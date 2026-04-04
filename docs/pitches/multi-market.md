---
tags:
  - pitch
  - markets
  - corners
  - cards
  - asian-handicap
  - correct-score
  - betfair
---

# Pitch — Multi-Market Prediction (v1.0.0)

## Problema

Nosso modelo só prevê 2 dos ~8 mercados da Betfair: resultado (1X2) e Over/Under 2.5. Estamos deixando na mesa:

1. **Placar Exato** — já calculamos (score_matrix) mas não mostramos
2. **Over/Under 0.5, 1.5, 3.5** — já calculamos mas só expomos o 2.5
3. **BTTS** — já calculamos mas não usamos pra bets
4. **Asian Handicap** — derivável dos dados que temos
5. **Escanteios** — Sofascore tem dados, precisa de Poisson separado
6. **Cartões** — Sofascore tem dados, precisa de Poisson + fator árbitro
7. **Half Time** — precisa Monte Carlo separado pro 1T
8. **Gol do Jogador** — precisa xG individual + lineup confirmada

Cada mercado adicional = mais oportunidades de value bet. A Starlizard (Tony Bloom, £600M/ano) usa modelos separados pra cada mercado.

Research: [[multi-market-prediction]]

## Solução

3 fases: P0 (expor o que temos), P1 (modelos novos), P2 (player props).

### Fase P0 — Expor mercados existentes (zero modelo novo)

Do Monte Carlo que já rodamos, derivar:

**Correct Score (Placar Exato):**
```python
# Já temos score_matrix: {"1x0": 0.15, "2x1": 0.12, ...}
# Expor top 10 placares com probabilidade
```

**Over/Under completo:**
```python
# Já temos over_05, over_15, over_25, over_35
# Expor todos + under correspondente
```

**BTTS:**
```python
# Já temos btts_prob
# Calcular btts_no = 1 - btts_prob
```

**Asian Handicap (derivado do score_matrix):**
```python
def calculate_asian_handicap(score_matrix: dict) -> dict:
    """Derivar probabilidades de handicap do score matrix."""
    # AH -0.5 home = P(home win) = sum onde home > away
    # AH -1.5 home = P(home win by 2+) = sum onde home > away + 1
    # AH +0.5 home = P(home win or draw)
    # AH -0.5 away = P(away win)
    # etc.
```

### Fase P1 — Novos modelos Poisson

**Escanteios (Corners Over/Under):**
```python
def predict_corners(
    home_corners_avg: float,    # média corners casa (últimos 6 jogos)
    away_corners_avg: float,    # média corners fora
    home_corners_against: float, # corners sofridos em casa
    away_corners_against: float,
) -> dict:
    """Poisson com λ = team_avg × opponent_factor."""
    lambda_home = home_corners_avg * (away_corners_against / league_avg)
    lambda_away = away_corners_avg * (home_corners_against / league_avg)
    # Monte Carlo: simular home_corners + away_corners
    # Over/Under 7.5, 8.5, 9.5, 10.5, 11.5
```

Dados necessários: escanteios por time por jogo (Sofascore tem).

**Cartões (Cards Over/Under):**
```python
def predict_cards(
    home_fouls_avg: float,      # faltas por jogo do mandante
    away_fouls_avg: float,      # faltas do visitante
    referee_cards_avg: float,   # cartões médios deste árbitro
    is_derby: bool,             # clássico?
) -> dict:
    """Zero-Inflated Poisson ajustado pelo árbitro."""
    base_lambda = (home_fouls_avg + away_fouls_avg) * referee_card_rate
    if is_derby:
        base_lambda *= 1.2  # +20% em clássicos
    # Monte Carlo: simular total de cartões
    # Over/Under 2.5, 3.5, 4.5, 5.5
```

Dados necessários: faltas e cartões por time (já temos), árbitro do jogo (Sofascore tem, precisa ingerir).

**Half Time Result:**
```python
def predict_half_time(home_xg: float, away_xg: float) -> dict:
    """Monte Carlo com λ_HT ≈ 45% do λ_FT."""
    home_ht_xg = home_xg * 0.45
    away_ht_xg = away_xg * 0.45
    # Simular: P(home HT), P(draw HT), P(away HT)
    # HT/FT combinado: P(home/home), P(draw/home), etc.
```

### Fase P2 — Player Props

**Gol do Jogador X:**
```python
def predict_player_goal(player_xg_per90: float, expected_minutes: int) -> float:
    """P(gol) = 1 - e^(-xG_per90 × min/90)."""
```

Precisa: lineup confirmada (~1h antes), xG individual (já temos).

## Arquitetura

### Módulos afetados

| Módulo | Ação | Descrição |
|--------|------|-----------|
| `domain/match_predictor.py` | MODIFICAR | Adicionar `calculate_asian_handicap()` derivado do score_matrix |
| `domain/corners_predictor.py` | NOVO | Poisson pra escanteios |
| `domain/cards_predictor.py` | NOVO | ZIP pra cartões |
| `domain/markets.py` | NOVO | Agregar todos os mercados num dict unificado |
| `adapters/sofascore_provider.py` | MODIFICAR | Ingerir escanteios, cartões, faltas, árbitro |
| `adapters/postgres_repository.py` | MODIFICAR | Queries pra corners/cards/referee stats |
| `adapters/orm.py` | MODIFICAR | Campos novos em player_match_metrics (ou tabela separada) |
| `use_cases/predict_all.py` | MODIFICAR | Rodar todos os preditores |
| `api.py` | MODIFICAR | Retornar todos os mercados |
| `frontend/` | MODIFICAR | Tabs por mercado em cada card |

### Schema

```sql
-- Campos extras no player_match_metrics (Sofascore já retorna)
-- corners, cards_yellow, cards_red, fouls já existem parcialmente

-- Nova tabela pra stats de árbitro
CREATE TABLE IF NOT EXISTS referee_stats (
    referee_name VARCHAR PRIMARY KEY,
    matches_officiated INTEGER,
    avg_cards_per_match REAL,
    avg_fouls_per_match REAL,
    avg_corners_per_match REAL,
    last_updated VARCHAR
);

-- Nova tabela pra match-level stats (corners, cards totais)
CREATE TABLE IF NOT EXISTS match_stats (
    match_id INTEGER PRIMARY KEY,
    home_corners INTEGER,
    away_corners INTEGER,
    home_yellow_cards INTEGER,
    away_yellow_cards INTEGER,
    home_red_cards INTEGER,
    away_red_cards INTEGER,
    home_fouls INTEGER,
    away_fouls INTEGER,
    referee_name VARCHAR,
    ht_home_score INTEGER,
    ht_away_score INTEGER
);
```

### Infra (K8s)

Sem mudanças.

## Escopo

### Fase P0 — Dentro do Escopo (sprint 1)

- [ ] `domain/markets.py` — agregar score_matrix → correct score, asian handicap, all O/U, BTTS
- [ ] `api.py` — retornar `markets` dict completo em cada prediction
- [ ] Frontend — tabs/seções: "Resultado", "Gols", "Placar Exato", "Handicap"
- [ ] Value bets pra todos os mercados existentes (não só h2h e totals)

### Fase P1 — Dentro do Escopo (sprint 2)

- [ ] Ingerir dados extras do Sofascore: escanteios, cartões, faltas, árbitro por jogo
- [ ] Tabelas `match_stats` e `referee_stats`
- [ ] `domain/corners_predictor.py` — Poisson pra escanteios
- [ ] `domain/cards_predictor.py` — ZIP pra cartões
- [ ] Half Time prediction no Monte Carlo
- [ ] Frontend — tabs: "Escanteios", "Cartões", "Intervalo"

### Fase P2 — Dentro do Escopo (sprint 3)

- [ ] Gol do Jogador X (xG individual + lineup)
- [ ] Frontend — tab "Jogador"

### Fora do Escopo

- "Criar Aposta" / Bet Builder (combinação custom) — complexo demais
- "Substituição Segura" — depende de decisão do treinador
- Live/in-play predictions
- Mercados de time (handicap asiático de escanteios)

## Research Necessária

- [x] Mercados Betfair e como prever — [[multi-market-prediction]]
- [ ] Validar que Sofascore retorna escanteios e árbitro pra Brasileirão
- [ ] Calibrar λ de escanteios no Brasileirão (média ~10/jogo?)
- [ ] Calibrar λ de cartões no Brasileirão + variância por árbitro
- [ ] Testar: HT goals = 45% ou diferente no Brasileirão?

## Estratégia de Testes

### Unitários (domain — zero mocks)
- `calculate_asian_handicap`: score_matrix conhecida → handicaps corretos
- `corners_predictor`: λ=5 home + λ=5 away → Over 9.5 ~50%
- `cards_predictor`: referee com 5 cards/game → Over 3.5 alto

### Manual
- Comparar Asian Handicap calculado com odds Betfair
- Comparar corners prediction com linhas de casas de apostas

## Critérios de Sucesso

- [ ] P0: todos os mercados deriváveis expostos no frontend
- [ ] P0: value bets identificadas em correct score e asian handicap (não só h2h)
- [ ] P1: corners prediction com accuracy > 50% no Over/Under 9.5
- [ ] P1: cards prediction com accuracy > 50% no Over/Under 3.5
- [ ] Cada mercado tem calibração validada contra odds Betfair
