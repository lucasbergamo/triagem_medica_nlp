"""Exportação ONNX do classificador — TF-IDF permanece em Python (ver docs/latencia.md).

Duas tentativas anteriores a este desenho, ambas medidas contra o test real (1.685 amostras),
não só supostas:

1. **Pipeline inteiro** (TF-IDF + LogisticRegression) pelo `skl2onnx.convert_sklearn`: o
   conversor de `TfidfVectorizer` recusa `strip_accents='unicode'` — contornável fazendo o
   strip de acentos em Python antes do ONNX Runtime — mas o tokenizador resultante diverge do
   sklearn em ngrams e casos de borda (números, hífens): até 12 predições de classe trocavam
   e a diferença máxima de probabilidade passava de 0,10, muito acima da tolerância de
   equivalência. É o risco nº2 do desenho original se confirmando — a conversão do
   vetorizador não é confiável o bastante para produção.
2. **Só o classificador**, pelo conversor padrão do skl2onnx (`LinearClassifier`, domínio
   `ai.onnx.ml`): equivalência perfeita (diff ~1e-7), mas `quantize_dynamic` não enxerga
   nenhum nó quantizável nesse domínio — o artefato "int8" saía do mesmo tamanho do fp32, sem
   ganho nenhum.

A solução adotada: reconstruir a etapa do classificador manualmente como `Gemm` (pesos) +
`Softmax` (probabilidades) — dois operadores do domínio padrão `ai.onnx`, matematicamente
idênticos à regressão logística multinomial do scikit-learn (`softmax(X @ coef_.T +
intercept_)`, diff numérica 0.0 nos testes) e, ao contrário do `LinearClassifier`, alvo válido
da quantização dinâmica do ONNX Runtime. `Gemm` também é a leitura mais literal do que já foi
dito sobre a escolha do modelo: "converte a inferência para um GEMM".

O TF-IDF continua em `sklearn`, dentro do `OnnxPredictor` (`src/models/predictor.py`) — os
backends `onnx`/`onnx-int8` carregam o mesmo `pipeline.joblib` do backend `sklearn` só para a
etapa de vetorização.
"""

import numpy as np
import onnx
from joblib import load
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score

from src.data.features import load_gold
from src.models.predictor import NOME_ARTEFATO_ONNX, NOME_ARTEFATO_ONNX_INT8, NOME_ARTEFATO_SKLEARN
from src.utils.config import MODELS_STAGING_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# fp32: a exportação deve ser matematicamente equivalente ao sklearn.
TOLERANCIA_PROBA_FP32 = 1e-4

# int8: quantização troca precisão por tamanho/latência por desenho — exigir "idêntico ao
# fp32" anularia o próprio ponto de quantizar. O gate é degradação de macro-F1 no test real,
# não predição idêntica (medido: degradação de 0,0009 no corpus atual — bem abaixo do piso).
TOLERANCIA_DEGRADACAO_MACRO_F1_INT8 = 0.02


def _construir_grafo_classificador(pipeline) -> onnx.ModelProto:
    """Gemm + Softmax equivalentes à LogisticRegression multinomial do pipeline staged."""
    clf = pipeline.named_steps["clf"]
    n_features = len(pipeline.named_steps["tfidf"].get_feature_names_out())
    n_classes = len(clf.classes_)

    coef = clf.coef_.astype(np.float32)  # (n_classes, n_features)
    intercept = clf.intercept_.astype(np.float32)  # (n_classes,)

    peso = numpy_helper.from_array(coef.T.copy(), name="peso")  # (n_features, n_classes)
    vies = numpy_helper.from_array(intercept, name="vies")

    entrada = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])
    saida = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, n_classes])

    gemm = helper.make_node(
        "Gemm", inputs=["input", "peso", "vies"], outputs=["logits"], alpha=1.0, beta=1.0
    )
    softmax = helper.make_node("Softmax", inputs=["logits"], outputs=["probabilities"], axis=1)

    grafo = helper.make_graph(
        [gemm, softmax],
        "classificador_urgencia",
        [entrada],
        [saida],
        initializer=[peso, vies],
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 18)])
    modelo.ir_version = 9
    onnx.checker.check_model(modelo)
    return modelo


def _inferir_onnx(onnx_bytes: bytes, x: np.ndarray) -> np.ndarray:
    sessao = InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    entrada = sessao.get_inputs()[0].name
    (probas,) = sessao.run(None, {entrada: x})
    return probas


def _validar_fp32(pipeline, x: np.ndarray, onnx_bytes: bytes) -> None:
    """Equivalência estrita: o fp32 deve reproduzir o sklearn dentro de `TOLERANCIA_PROBA_FP32`."""
    probas_onnx = _inferir_onnx(onnx_bytes, x)
    probas_sklearn = pipeline.named_steps["clf"].predict_proba(x)
    classes = pipeline.named_steps["clf"].classes_

    preds_onnx = classes[probas_onnx.argmax(axis=1)]
    preds_sklearn = classes[probas_sklearn.argmax(axis=1)]

    diff_maxima = float(np.max(np.abs(probas_onnx - probas_sklearn)))
    n_divergencias = int(np.sum(preds_onnx != preds_sklearn))

    logger.info(
        "validacao_onnx_fp32", diff_maxima=diff_maxima, divergencias=n_divergencias, n=len(x)
    )

    if diff_maxima > TOLERANCIA_PROBA_FP32 or n_divergencias > 0:
        raise ValueError(
            f"ONNX (fp32) diverge do sklearn além da tolerância: diff máxima "
            f"{diff_maxima:.6f} (tolerância {TOLERANCIA_PROBA_FP32}), {n_divergencias} "
            f"predições trocadas em {len(x)} amostras."
        )


def _validar_int8(pipeline, x: np.ndarray, y: list[str], onnx_bytes: bytes) -> dict:
    """Gate de degradação — não de equivalência. Ver `TOLERANCIA_DEGRADACAO_MACRO_F1_INT8`."""
    probas_onnx = _inferir_onnx(onnx_bytes, x)
    classes = pipeline.named_steps["clf"].classes_
    preds_onnx = classes[probas_onnx.argmax(axis=1)]
    preds_sklearn = pipeline.named_steps["clf"].predict(x)

    macro_f1_onnx = float(f1_score(y, preds_onnx, average="macro"))
    macro_f1_sklearn = float(f1_score(y, preds_sklearn, average="macro"))
    degradacao = macro_f1_sklearn - macro_f1_onnx
    n_divergencias = int(np.sum(preds_onnx != preds_sklearn))

    logger.info(
        "validacao_onnx_int8",
        macro_f1_int8=round(macro_f1_onnx, 4),
        macro_f1_sklearn=round(macro_f1_sklearn, 4),
        degradacao=round(degradacao, 4),
        divergencias=n_divergencias,
        n=len(y),
    )

    if degradacao > TOLERANCIA_DEGRADACAO_MACRO_F1_INT8:
        raise ValueError(
            f"ONNX (int8) degrada macro-F1 além da tolerância: {macro_f1_sklearn:.4f} → "
            f"{macro_f1_onnx:.4f} (degradação {degradacao:.4f}, tolerância "
            f"{TOLERANCIA_DEGRADACAO_MACRO_F1_INT8})."
        )
    return {
        "macro_f1_int8": round(macro_f1_onnx, 4),
        "macro_f1_sklearn": round(macro_f1_sklearn, 4),
        "divergencias_int8": n_divergencias,
    }


def _dedupe_opset(modelo: onnx.ModelProto) -> None:
    """Remove entradas duplicadas de `opset_import` por domínio.

    O grafo construído à mão por este módulo já sai limpo — esta função existe porque o
    conversor padrão do skl2onnx (usado na primeira tentativa, ver docstring do módulo) grava
    a mesma entrada `("", 22)` duas vezes, e `quantize_dynamic` recusa processar o grafo
    quando encontra mais de uma entrada de domínio `ai.onnx` (erro observado: 'Failed to find
    proper ai.onnx domain'). Mantida como rede de segurança caso o conversor padrão volte a
    ser usado em algum backend futuro.
    """
    vistos: dict[str, onnx.OperatorSetIdProto] = {}
    for op in modelo.opset_import:
        atual = vistos.get(op.domain)
        if atual is None or op.version > atual.version:
            vistos[op.domain] = op
    del modelo.opset_import[:]
    modelo.opset_import.extend(vistos.values())


def run() -> dict:
    """Exporta o classificador staged para ONNX (fp32 e int8), valida contra o sklearn com
    dados reais de test, e grava os dois artefatos em `models/staging/`.

    Não promove nada — só `src.models.registry.promover()` escreve em `models/current/`.
    """
    pipeline = load(MODELS_STAGING_DIR / NOME_ARTEFATO_SKLEARN)
    _train, _val, test, _metadata = load_gold()
    textos = test["texto"].tolist()
    y = test["urgencia"].tolist()
    x = pipeline.named_steps["tfidf"].transform(textos).toarray().astype(np.float32)

    modelo_onnx = _construir_grafo_classificador(pipeline)
    _dedupe_opset(modelo_onnx)
    onnx_bytes = modelo_onnx.SerializeToString()
    _validar_fp32(pipeline, x, onnx_bytes)

    path_onnx = MODELS_STAGING_DIR / NOME_ARTEFATO_ONNX
    path_onnx.write_bytes(onnx_bytes)

    path_int8 = MODELS_STAGING_DIR / NOME_ARTEFATO_ONNX_INT8
    quantize_dynamic(str(path_onnx), str(path_int8), weight_type=QuantType.QInt8)
    resultado_int8 = _validar_int8(pipeline, x, y, path_int8.read_bytes())

    resultado = {
        "onnx_bytes": len(onnx_bytes),
        "onnx_int8_bytes": path_int8.stat().st_size,
        "n_amostras_validadas": len(y),
        **resultado_int8,
    }
    logger.info("exportacao_onnx_concluida", **resultado)
    return resultado


def main() -> None:
    run()


if __name__ == "__main__":
    main()
