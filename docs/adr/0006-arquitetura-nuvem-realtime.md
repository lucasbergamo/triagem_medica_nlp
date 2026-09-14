# ADR-0006 — Arquitetura de nuvem: inferência em tempo real, não batch nem serverless

## Status

Aceito.

## Contexto

A triagem existe para ordenar a fila de atendimento **enquanto o paciente está esperando** —
esse propósito, não uma preferência de implementação, é o que decide o padrão de arquitetura
antes de qualquer detalhe técnico. Três padrões de inferência em nuvem foram comparados:

| Padrão | Avaliação |
|---|---|
| **Batch** | Reprovado. Um lote processado periodicamente entrega a classificação depois que a decisão de atendimento já foi tomada — o resultado chega tarde demais para servir ao propósito do produto |
| **Serverless** (função sob demanda) | Reprovado. O cold start de uma função fria — dezenas de milissegundos — domina o tempo de resposta quando a inferência em si custa menos de 1 ms (`docs/latencia.md`): o overhead de infraestrutura passaria a ser o gargalo, exatamente o que a otimização de latência deste projeto existe para eliminar |
| **Real-time (API síncrona)** | **Escolhido.** Um serviço sempre ativo, com o modelo já carregado em memória, responde no tempo da requisição — o único padrão compatível com um SLO de latência apertado (p95 abaixo de 100 ms fim a fim) |

## Decisão

A API roda como serviço containerizado sempre ativo (não uma função por requisição), com o
modelo carregado uma única vez no `lifespan` (ADR-0003) e a separação `/health`/`/ready` como
contrato de saúde para o orquestrador. O alvo de implantação segue o mesmo raciocínio: ECS
Fargate atrás de um balanceador que só recebe tráfego depois que `/ready` confirma o modelo
carregado, nunca uma função efêmera por requisição, com exposição via API Gateway → VPC Link →
NLB interno → Fargate em subnet privada — o mesmo padrão de camadas de segurança de um projeto
anterior desta pós-graduação, reaproveitado aqui.

**O provisionamento real não faz parte desta entrega.** O critério de avaliação deste projeto
cobre a decisão de arquitetura em texto, não a implantação em si — o desenho acima documenta a
decisão e o caminho teria custo de implementação (Terraform, ECR, EventBridge para o retreino
agendado) proporcional ao de um módulo novo, não de um ajuste de configuração. Os sete
princípios abaixo, já aplicados no código local, existem justamente para que esse caminho, se
percorrido depois, seja curto:

1. **12-factor** — toda configuração via ambiente, lida por Pydantic Settings; zero caminho
   absoluto, zero `localhost` fixo no código.
2. **`ModelStore` abstrato** (`src/models/store.py`) — `LocalModelStore` hoje; um backend de
   object store entraria como segunda implementação da mesma interface, sem tocar no resto do
   código.
3. **Imagem sem estado** — nada é escrito em disco local em runtime; reciclar o container não
   perde nada.
4. **`/health` ≠ `/ready`** — contrato que ALB/ECS exigem separado (ADR-0003).
5. **Log estruturado em stdout** (`structlog`) — um coletor de logs gerenciado faz parsing sem
   adaptador.
6. **`/metrics` em formato Prometheus/OpenMetrics padrão** — um serviço gerenciado de métricas
   faz scrape sem adaptador.
7. **Retreino como grafo de dependências, não script solto** — a DAG do Airflow (ADR-0005) já
   expressa o pipeline `dados → treino → avaliação (gate) → exportação → promoção → benchmark`
   de forma que um agendador gerenciado (ex.: execução programada de container) só precisa
   disparar, não reimplementar.

## Consequências

- Nenhum recurso de nuvem é provisionado ou cobrado por este projeto.
- A decisão documentada aqui é verificável no comportamento do código local: os sete princípios
  acima não são promessas, são o que a API e a DAG já fazem hoje.
- Se o provisionamento for retomado no futuro, o trabalho adicional é escrever a infraestrutura
  declarativa em cima do desenho já pronto, não redesenhar a aplicação.
