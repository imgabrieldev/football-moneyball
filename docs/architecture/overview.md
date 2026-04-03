---
tags:
  - architecture
  - overview
---

# Arquitetura — Football Moneyball

## Visão Geral

Sistema de analytics de futebol que combina dados abertos do StatsBomb com análise estatística avançada, grafos de rede e busca vetorial para quantificar jogadores e identificar padrões táticos.

## Stack

- **Linguagem:** Python 3.12+
- **Dados:** StatsBomb Open Data (via statsbombpy)
- **Banco:** PostgreSQL 16 + pgvector (busca vetorial)
- **Infra:** Minikube, Kustomize (sem Helm)
- **CLI:** Typer + Rich
- **ML:** scikit-learn (PCA, KMeans, Ridge)
- **Grafos:** networkx
- **Viz:** matplotlib + mplsoccer

## Módulos

```
football_moneyball/
├── db.py               # ORM + pgvector queries (camada de dados)
├── player_metrics.py   # Extração de ~30 métricas do StatsBomb
├── network_analysis.py # Rede de passes (grafo networkx)
├── player_embeddings.py # Embeddings PCA + clustering + pgvector
├── rapm.py             # RAPM via Ridge regression
├── viz.py              # Visualizações (campo, radar, heatmap, RAPM)
├── export.py           # Relatórios markdown/JSON
└── cli.py              # CLI Typer (8 comandos)
```

## Fluxo de Dados

```
StatsBomb API → player_metrics.py → PostgreSQL (player_match_metrics)
                                   ↓
             network_analysis.py → PostgreSQL (pass_networks)
                                   ↓
            player_embeddings.py → PostgreSQL (player_embeddings + pgvector index)
                                   ↓
                        rapm.py → PostgreSQL (stints)
                                   ↓
                  viz.py / export.py / cli.py (apresentação)
```

## Banco de Dados

5 tabelas + 1 índice HNSW:

| Tabela | Chave Primária | Propósito |
|--------|---------------|-----------|
| `matches` | `match_id` | Cache de metadados das partidas |
| `player_match_metrics` | `(match_id, player_id)` | ~30 métricas por jogador por partida |
| `pass_networks` | `(match_id, passer_id, receiver_id)` | Arestas do grafo de passes |
| `player_embeddings` | `(player_id, season)` | Vetores de estilo (vector(16)) + arquétipo |
| `stints` | `(match_id, stint_number)` | Períodos com mesmos 22 em campo |

### pgvector

- Coluna `embedding vector(16)` na tabela `player_embeddings`
- Índice HNSW com `vector_cosine_ops` para approximate nearest neighbors
- Operadores: `<=>` (cosine distance), `<->` (L2), `<#>` (inner product)

## Infra (Minikube)

```
k8s/
├── namespace.yaml      # football-moneyball
├── secret.yaml         # POSTGRES_USER/PASSWORD/DB
├── configmap.yaml      # init.sql (schema + vector extension)
├── pvc.yaml            # 1Gi storage
├── deployment.yaml     # pgvector/pgvector:pg16
├── service.yaml        # ClusterIP:5432
└── kustomization.yaml  # referencia todos os resources
```

CLI roda local, banco no Minikube via `kubectl port-forward`.
