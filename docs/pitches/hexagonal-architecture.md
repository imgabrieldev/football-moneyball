---
tags:
  - pitch
  - architecture
  - clean-code
  - refactor
  - hexagonal
---

# Pitch — Hexagonal Architecture & Clean Code (v0.3.0)

## Problema

O codebase cresceu de 7 módulos (v0.1.0) para 11 módulos + scripts + testes (v0.2.0), e tem problemas estruturais que vão escalar mal:

1. **Acoplamento com providers externos** — `player_metrics.py`, `pressing.py`, `network_analysis.py`, `rapm.py` e `possession_value.py` todos importam `statsbombpy` diretamente. `scripts/ingest_sofascore.py` usa `requests` + API Sofascore. Adicionar um novo provider (API-Football, Opta) requer duplicar lógica de domínio.

2. **Acoplamento com infraestrutura** — `player_embeddings.py`, `rapm.py` e `export.py` importam SQLAlchemy diretamente e fazem queries SQL raw. Domain logic (PCA, clustering, Ridge regression) está misturada com persistence logic (session, upsert, text queries).

3. **Módulos com múltiplas responsabilidades** — `player_metrics.py` faz extração de dados StatsBomb + cálculo de métricas + extração de posições. `cli.py` faz orquestração + formatação + lógica de negócio (xG contribution, aggregation).

4. **Sem inversão de dependência** — Módulos de alto nível (analysis, embeddings) dependem diretamente de módulos de baixo nível (db, statsbombpy). Não há interfaces/protocolos entre camadas.

5. **Testabilidade limitada** — Testes atuais mockam `sb.events()` porque não há abstração sobre o provider. Testes de integração requerem PostgreSQL real sem alternativa in-memory.

6. **Script Sofascore solto** — `scripts/ingest_sofascore.py` duplica lógica de conversão e persistência que já existe nos módulos principais, sem compartilhar interfaces.

Research: [[market-standard-analytics]], [[socceraction-statsbomb-api]], [[brasileirao-data-sources]]

## Solução

Refatorar para **Hexagonal Architecture (Ports & Adapters)** com 3 camadas claras:

```
┌─────────────────────────────────────────────────┐
│                  Adapters (IN)                   │
│  CLI (typer)  │  Scripts  │  API (futuro)        │
├─────────────────────────────────────────────────┤
│                  Use Cases                       │
│  analyze_match │ analyze_season │ compare │ ...  │
├─────────────────────────────────────────────────┤
│                   Domain                         │
│  metrics │ pressing │ xT │ embeddings │ rapm     │
│  models (Player, Match, Team, Action)            │
├─────────────────────────────────────────────────┤
│                  Ports (interfaces)               │
│  DataProvider │ Repository │ Visualizer           │
├─────────────────────────────────────────────────┤
│                 Adapters (OUT)                    │
│  StatsBomb │ Sofascore │ PostgreSQL │ Matplotlib │
└─────────────────────────────────────────────────┘
```

**Princípios:**
- Domain layer não importa nada externo (nem pandas, nem sqlalchemy, nem statsbombpy)
- Domain recebe e retorna dataclasses/dicts puros
- Ports definem interfaces (Protocol classes) para data access e persistence
- Adapters implementam os ports (StatsBomb, Sofascore, PostgreSQL, matplotlib)
- Use cases orquestram domain + ports
- Dependency Injection via construtor nos use cases

## Arquitetura

### Estrutura de diretórios proposta

```
football_moneyball/
├── domain/                    # Lógica de negócio pura — ZERO dependências externas
│   ├── models.py              # Dataclasses: Player, Match, MatchMetrics, ActionValue, etc.
│   ├── metrics.py             # Cálculo de métricas a partir de events (recebe listas/dicts)
│   ├── pressing.py            # PPDA, success rate, counter-pressing (lógica pura)
│   ├── possession_value.py    # xT model (numpy apenas — é math puro)
│   ├── embeddings.py          # PCA, clustering, archetipos (sklearn)
│   ├── rapm.py                # Ridge regression, splints, SPM prior (sklearn + numpy)
│   ├── network.py             # Grafo de passes, centralidade (networkx)
│   └── constants.py           # POSITION_GROUP_MAP, grid sizes, thresholds
│
├── ports/                     # Interfaces (Protocol classes)
│   ├── data_provider.py       # Protocol: get_events(), get_lineups(), get_competitions()
│   ├── repository.py          # Protocol: save_match(), get_player_metrics(), find_similar()
│   └── visualizer.py          # Protocol: plot_pass_network(), plot_radar(), etc.
│
├── adapters/                  # Implementações dos ports
│   ├── statsbomb_provider.py  # Implementa DataProvider via statsbombpy
│   ├── sofascore_provider.py  # Implementa DataProvider via API Sofascore
│   ├── postgres_repository.py # Implementa Repository via SQLAlchemy + pgvector
│   ├── orm.py                 # ORM models (SQLAlchemy) — isolado do domain
│   └── matplotlib_viz.py     # Implementa Visualizer via matplotlib + mplsoccer
│
├── use_cases/                 # Orquestração — conecta domain + ports
│   ├── analyze_match.py       # UseCase: extrair, calcular, persistir, exibir
│   ├── analyze_season.py      # UseCase: processar temporada inteira
│   ├── compare_players.py     # UseCase: comparar dois jogadores
│   ├── find_similar.py        # UseCase: busca por similaridade
│   ├── generate_report.py     # UseCase: scout report
│   └── ingest_external.py     # UseCase: ingestão de fonte externa (Sofascore)
│
├── cli.py                     # Adapter IN: CLI Typer (thin layer → use cases)
└── config.py                  # DATABASE_URL, feature flags, provider selection
```

### Módulos afetados

**TODOS os módulos serão refatorados.** Mapeamento old → new:

| Módulo atual | Destino | O que muda |
|---|---|---|
| `db.py` (ORM + queries + session) | `adapters/orm.py` + `adapters/postgres_repository.py` | Split: ORM models separados de query logic |
| `player_metrics.py` (StatsBomb + cálculos) | `domain/metrics.py` + `adapters/statsbomb_provider.py` | Split: extração de dados vs cálculo de métricas |
| `pressing.py` (StatsBomb + cálculos) | `domain/pressing.py` + provider | Remove import de statsbombpy |
| `possession_value.py` (StatsBomb + xT) | `domain/possession_value.py` + provider | xT vira math puro, dados vêm via port |
| `network_analysis.py` (StatsBomb + networkx) | `domain/network.py` + provider | Grafo recebe dados, não busca |
| `player_embeddings.py` (sklearn + SQLAlchemy) | `domain/embeddings.py` + repository | Remove SQLAlchemy, recebe dados via port |
| `rapm.py` (StatsBomb + sklearn + SQLAlchemy) | `domain/rapm.py` + provider + repository | Remove ambos, recebe dados via port |
| `viz.py` | `adapters/matplotlib_viz.py` | Renomear, implementar port |
| `export.py` (SQLAlchemy + formatação) | `use_cases/generate_report.py` + repository | Report logic via use case |
| `cli.py` (tudo) | `cli.py` (thin) + `use_cases/*.py` | CLI vira fina: parseia args → chama use case |
| `scripts/ingest_sofascore.py` | `adapters/sofascore_provider.py` + `use_cases/ingest_external.py` | Provider implementa interface, use case orquestra |

### Domain Models (`domain/models.py`)

```python
from dataclasses import dataclass, field

@dataclass
class Player:
    player_id: int
    name: str
    team: str
    position_group: str = "MID"  # GK, DEF, MID, FWD

@dataclass
class MatchInfo:
    match_id: int
    competition: str
    season: str
    match_date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int

@dataclass
class PlayerMatchMetrics:
    player: Player
    match_id: int
    minutes_played: float = 0.0
    goals: int = 0
    assists: int = 0
    xg: float = 0.0
    xa: float = 0.0
    # ... todos os ~45 campos como atributos tipados
    
@dataclass
class ActionValue:
    event_index: int
    player: Player
    action_type: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    xt_value: float | None = None
    vaep_value: float | None = None

@dataclass
class PressingProfile:
    team: str
    ppda: float
    pressing_success_rate: float
    counter_pressing_fraction: float
    high_turnovers: int
    shot_ending_high_turnovers: int
    zones: list[float] = field(default_factory=lambda: [0.0] * 6)
```

### Ports (`ports/data_provider.py`)

```python
from typing import Protocol

class DataProvider(Protocol):
    """Interface para fontes de dados de futebol."""
    
    def get_match_events(self, match_id: int) -> list[dict]:
        """Retorna eventos de uma partida em formato normalizado."""
        ...
    
    def get_lineups(self, match_id: int) -> dict[str, list[dict]]:
        """Retorna lineups com posições."""
        ...
    
    def get_competitions(self) -> list[dict]:
        """Lista competições disponíveis."""
        ...
    
    def get_matches(self, competition_id: int, season_id: int) -> list[dict]:
        """Lista partidas de uma competição/temporada."""
        ...
```

```python
class Repository(Protocol):
    """Interface para persistência de dados."""
    
    def save_match(self, match: MatchInfo) -> None: ...
    def save_player_metrics(self, metrics: list[PlayerMatchMetrics], match_id: int) -> None: ...
    def get_player_metrics(self, player_name: str, season: str | None) -> list[PlayerMatchMetrics]: ...
    def find_similar_players(self, player_name: str, season: str, limit: int) -> list[dict]: ...
    # ...
```

### Schema

**Sem mudanças no PostgreSQL.** As tabelas e colunas ficam idênticas — apenas o código que interage com elas muda de lugar (de `db.py` para `adapters/postgres_repository.py`).

### Infra (K8s)

Sem mudanças. Manifestos Kubernetes ficam iguais.

## Escopo

### Dentro do Escopo

- [ ] Criar `domain/models.py` com dataclasses para todas as entidades
- [ ] Criar `domain/constants.py` com POSITION_GROUP_MAP, thresholds, grid sizes
- [ ] Criar `ports/data_provider.py` com Protocol DataProvider
- [ ] Criar `ports/repository.py` com Protocol Repository
- [ ] Criar `ports/visualizer.py` com Protocol Visualizer
- [ ] Criar `adapters/statsbomb_provider.py` — implementa DataProvider
- [ ] Criar `adapters/sofascore_provider.py` — implementa DataProvider (migrar de scripts/)
- [ ] Criar `adapters/orm.py` — mover ORM models de db.py
- [ ] Criar `adapters/postgres_repository.py` — implementa Repository
- [ ] Criar `adapters/matplotlib_viz.py` — implementa Visualizer (migrar de viz.py)
- [ ] Refatorar `domain/metrics.py` — remover statsbombpy, receber events como list[dict]
- [ ] Refatorar `domain/pressing.py` — remover statsbombpy
- [ ] Refatorar `domain/possession_value.py` — remover statsbombpy
- [ ] Refatorar `domain/network.py` — remover statsbombpy
- [ ] Refatorar `domain/embeddings.py` — remover SQLAlchemy
- [ ] Refatorar `domain/rapm.py` — remover statsbombpy e SQLAlchemy
- [ ] Criar use cases: analyze_match, analyze_season, compare_players, find_similar, generate_report, ingest_external
- [ ] Refatorar `cli.py` — thin layer que parseia args e chama use cases
- [ ] Criar `config.py` — configuration centralizada com DI
- [ ] Migrar testes para nova estrutura
- [ ] Manter 100% backward compatibility nos comandos CLI
- [ ] `pyproject.toml` — atualizar package discovery

### Fora do Escopo

- Novas features ou métricas (isso é refactor puro)
- Mudanças no schema PostgreSQL
- Mudanças nos manifests K8s
- Novos comandos CLI
- Performance optimization
- Async/await
- API REST (será pitch separado quando tiver frontend)

## Research Necessária

- [ ] Validar que Protocol classes do Python funcionam como interfaces para DI sem framework
- [ ] Confirmar que sklearn/numpy são aceitáveis no domain layer (são math libraries, não infra)

## Estratégia de Testes

### Unitários (domain layer)
- Testes de `domain/metrics.py` com dicts puros (sem mock de StatsBomb)
- Testes de `domain/pressing.py` com eventos fabricados (sem mock)
- Testes de `domain/possession_value.py` — xT com arrays numpy
- Testes de `domain/embeddings.py` — PCA/clustering com DataFrames sintéticos
- Testes de `domain/rapm.py` — Ridge com matrizes numpy

### Unitários (adapters)
- `adapters/statsbomb_provider.py` — mock statsbombpy, verificar conversão
- `adapters/sofascore_provider.py` — mock requests, verificar conversão
- `adapters/postgres_repository.py` — mock session, verificar queries

### Integração
- Use cases com mocks dos ports
- End-to-end: CLI → use case → domain → adapters (com PG de teste)

### Regressão
- Todos os 42 testes existentes devem continuar passando (adaptados)
- CLI commands produzem mesmos outputs

## Critérios de Sucesso

- [ ] `domain/` não importa statsbombpy, sqlalchemy, requests, typer, rich, matplotlib
- [ ] `domain/` pode ser testado sem banco de dados e sem API externa
- [ ] Adicionar novo DataProvider (ex: API-Football) requer apenas 1 arquivo novo em `adapters/`
- [ ] Todos os 42+ testes passam
- [ ] Todos os 8 comandos CLI funcionam identicamente
- [ ] `python3 -m py_compile` passa em todos os módulos
- [ ] Nenhuma mudança no schema PostgreSQL
- [ ] Nenhuma mudança nos manifests K8s
