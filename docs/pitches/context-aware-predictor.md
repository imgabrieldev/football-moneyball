---
tags:
  - pitch
  - prediction
  - features
  - context
  - coach
  - injuries
  - ml
---

# Pitch — Context-Aware Predictor (v1.6.0)

## Problema

Nosso modelo stats-only (v1.5.x) olha só números agregados dos últimos 5 jogos. **Ignora contexto que apostadores profissionais usam intensamente.**

**Caso real descoberto (Vasco × Botafogo, 04/04/2026):**
- Modelo previu Vasco 77% vencendo
- Mas contexto real é ainda mais favorável ao Vasco:
  - **Vasco sob técnico novo (Renato Gaúcho)** — 5 jogos invicto, 73% aproveitamento
  - **Botafogo com técnico INTERINO** (historicamente -20% performance)
  - **4 titulares do Botafogo desfalcados** (Joaquín Correa, Chris Ramos, Marçal, Kaio Pantaleão)
  - **Pior defesa da Série A** (Botafogo)
  - **São Januário lotado** (clássico carioca, 97-42 vitórias históricas)
- Contexto sugere **Vasco 80-85%** real. Modelo subestima.

**Caso inverso também acontece:**
- Time com "boa forma estatística" mas técnico demitido essa semana → modelo otimista demais
- Time com estrela lesionada (nossos player-aware λ ainda incluem ela) → superestima

**Gap:** modelo não sabe de:
1. Mudança de técnico recente
2. Técnico interino
3. Desfalques específicos no XI titular
4. Fadiga (fixture congestion)
5. Contexto de tabela (posição relativa)

Research: [[feature-rich-predictor]], [[comprehensive-prediction-models]]

## Solução

Adicionar **5 features contextuais** capturando info não-estatística. Tudo disponível via Sofascore API + scraping leve. Alimenta o mesmo XGBoost/GBR do v1.3.0/v1.5.0.

### Features novas (v1.6.0)

**1. Coach context**
```python
coach_change_recent: bool       # técnico mudou nos últimos 30 dias?
games_since_coach_change: int   # número de jogos sob técnico atual
coach_win_rate: float           # taxa de vitórias do técnico atual
is_interim_coach: bool          # técnico interino?
```

**2. Injuries/Availability**
```python
key_players_out: int            # desfalques entre top 3 por xG/90
starter_xi_available: bool      # XI titular provável disponível?
xg_contribution_missing: float  # xG/90 dos ausentes (sum)
```

**3. Fixture congestion (fadiga)**
```python
games_last_7d: int              # jogos nos últimos 7 dias
games_next_7d: int              # jogos nos próximos 7 dias
rest_days: int                  # já temos em v1.5.0 ✓
competition_count: int          # em quantas competições simultâneas
```

**4. League context**
```python
position_gap: int               # diferença de posição (home - away)
points_gap: int                 # diferença de pontos
home_relegation_pressure: bool  # zona Z4 ou próximo
away_relegation_pressure: bool
```

**5. Derby/rivalry**
```python
is_derby: bool                  # clássico estadual
h2h_home_advantage: float       # % vitórias casa histórico
```

Total: 15+ novas features → **FEATURE_DIM 24 → 40**.

## Arquitetura

### Módulos afetados

| Módulo | Ação | Descrição |
|--------|------|-----------|
| `domain/context_features.py` | NOVO | Lógica pura de feature engineering contextual |
| `domain/feature_engineering.py` | MODIFICAR | Expandir FEATURE_DIM 24→40 |
| `adapters/sofascore_provider.py` | MODIFICAR | `get_coach_info(team_id)`, `get_injuries(team_id)` |
| `adapters/postgres_repository.py` | MODIFICAR | `get_games_in_window(team, start, end)`, queries contextuais |
| `adapters/orm.py` | MODIFICAR | Nova tabela `team_coaches` (histórico), coluna `injured` em lineups |
| `use_cases/ingest_context.py` | NOVO | Ingere dados contextuais (técnicos + lesões) |
| `use_cases/predict_all.py` | MODIFICAR | Alimentar features contextuais no ML |

### Schema

```sql
-- Histórico de técnicos por time (quem treinou quando)
CREATE TABLE team_coaches (
    team VARCHAR(100),
    coach_name VARCHAR(100),
    start_date DATE,
    end_date DATE,              -- NULL = atual
    is_interim BOOLEAN DEFAULT false,
    games_coached INTEGER,
    wins INTEGER,
    draws INTEGER,
    losses INTEGER,
    PRIMARY KEY (team, start_date)
);

-- Lesões ativas (status por jogador por data)
CREATE TABLE player_injuries (
    player_id INTEGER,
    player_name VARCHAR(100),
    team VARCHAR(100),
    injury_type VARCHAR(50),
    reported_date DATE,
    expected_return DATE,        -- NULL se indefinido
    status VARCHAR(20),          -- 'out', 'doubt', 'returned'
    PRIMARY KEY (player_id, reported_date)
);

-- Classificação por rodada (pra position_gap, pressure zones)
CREATE TABLE league_standings (
    competition VARCHAR(100),
    season VARCHAR(20),
    round INTEGER,
    team VARCHAR(100),
    position INTEGER,
    points INTEGER,
    played INTEGER,
    wins INTEGER, draws INTEGER, losses INTEGER,
    goals_for INTEGER, goals_against INTEGER,
    PRIMARY KEY (competition, season, round, team)
);
```

### Infra (K8s)

**Novo CronJob:** `ingest-context` (diário, 6am)
- Busca coaches + injuries do Sofascore
- Atualiza standings da rodada atual

Sem mudanças no deployment API (só usa dados do PG).

## Escopo

### Dentro do Escopo

- [ ] Tabelas `team_coaches`, `player_injuries`, `league_standings`
- [ ] Sofascore adapter: `get_team_managers()`, `get_team_injuries()`, `get_standings()`
- [ ] `ingest_context.py` — puxa dados diariamente
- [ ] `context_features.py` — computa as 15 features
- [ ] Estender `feature_engineering.py` FEATURE_DIM 24→40
- [ ] Backfill histórico de técnicos + standings por rodada
- [ ] `predict_all.py` — alimenta features contextuais
- [ ] Frontend: badges "Técnico novo", "Interino", "X desfalques"
- [ ] CronJob `ingest-context` no K8s
- [ ] Tests: `test_domain_context_features.py`
- [ ] Retreinar modelos com 40 features

### Fora do Escopo

- Event-level data (WhoScored SPADL) — fica pra v1.7.0
- Weather data (chuva, calor) — marginal, ignorar por ora
- Player form (individual streaks) — reservado pra v1.8.0
- Manager tactical style (3-5-2 vs 4-3-3) — complexo, depois
- Sentiment analysis de mídia/torcida — low ROI

## Research Necessária

- [x] Context matters: Vasco × Botafogo caso real validado
- [x] Sofascore API validada:
  - `event/{id}/managers` → home/away manager por partida
  - `event/{id}/lineups.missingPlayers` → lesões/suspensões com reason code
  - `unique-tournament/.../standings/total` → classificação
- [x] Literatura "new manager bounce":
  - **Efeito modesto** (~10 jogos de boost) — *principalmente regressão à média*
  - Estudos: Bryson 2024 (Scottish J of Pol Econ), PMC 2021
- [x] Literatura "interim coaches":
  - **Não são necessariamente piores** — 2010 study mostra que podem performar MELHOR (motivação dos jogadores)
  - Implicação: usar `coach_win_rate` real, não flag negativa hardcoded
- [x] Literatura "fixture congestion":
  - Afeta **injury risk** mais que performance direta (squad rotation mitiga)
  - Meta-analysis: PMC 2021 confirma
- [x] Literatura "player impact":
  - Player Impact Metric (PIM) 2025 confirma: top-N players ausentes reduzem expected outcome
  - Transfermarkt value forte preditor (usaremos xG/90 como proxy)

## Estratégia de Testes

### Unitários (domain puro)

- `test_domain_context_features.py`:
  - `coach_change_recent`: True se < 30 dias
  - `key_players_out`: count correto do top 3 por xG/90
  - `fixture_congestion`: games em janela
  - `position_gap`: diferença numérica de rank
  - Fallback quando sem dados

### Integração (com PG)

- Ingest real coaches de Sofascore → tabela populada
- Query contextual retorna valores esperados pra Vasco × Botafogo
- Features rich (40-dim) × old (24-dim) reproduzem v1.5.0 quando context=neutral

### Manual

- Prever Vasco × Botafogo v1.5.0 vs v1.6.0 — comparar probabilidades
- Verificar que v1.6.0 dá Vasco 80%+ (contexto favorável)
- Comparar com odds Betfair (edge deveria aumentar se modelo acertar)

## Critérios de Sucesso

- [ ] 15+ features contextuais implementadas
- [ ] Coach history backfilled pros 20 times Brasileirão 2026
- [ ] Injuries backfilled via Sofascore
- [ ] Standings por rodada ingeridos
- [ ] Modelos retreinados com 40 features, MAE stable ou melhor
- [ ] **Brier < 0.19** em backtest time-split (era 0.21-0.22)
- [ ] Feature importance mostra ≥2 features contextuais no top-10
- [ ] Frontend mostra badges contextuais (técnico interino, desfalques, etc)
- [ ] Vasco × Botafogo previsto corretamente: Vasco 80%+

## Próximos Passos após v1.6.0

Se Brier < 0.19 ✓ → v1.7.0: Event data via WhoScored (xT, PPDA reais)
Se Brier ≥ 0.19 ✗ → investigar overfit, ajustar hyperparams, weighted feature groups
