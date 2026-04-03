---
tags:
  - research
  - socceraction
  - statsbomb
  - xT
  - VAEP
  - SPADL
  - pressing
  - positions
---

# Research — socceraction API & StatsBomb Event Schema

> Research date: 2026-04-03
> Sources: [listadas ao final]

## Context

Para implementar o pitch [[state-of-the-art-engine]], precisamos entender exatamente como usar a lib `socceraction` para xT/VAEP e quais campos do StatsBomb estão disponíveis para pressing metrics e position-aware embeddings.

---

## 1. socceraction — API Completa

### Instalação

```bash
pip install socceraction>=1.5
```

Lib MIT license, v1.5.3 (Dec 2024). **Não está em desenvolvimento ativo** — usada para reprodutibilidade de research. Suporta StatsBomb, Opta, Wyscout, Stats Perform, WhoScored.

### Pipeline: StatsBomb → SPADL → xT/VAEP

#### Step 1: Carregar dados StatsBomb

```python
from socceraction.data.statsbomb import StatsBombLoader

# Open data (sem credenciais)
SBL = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

# Listar
competitions = SBL.competitions()
games = SBL.games(competition_id=11, season_id=90)

# Carregar eventos de um jogo
events = SBL.events(game_id=3773585)

# Carregar lineup/players
players = SBL.players(game_id=3773585)
teams = SBL.teams(game_id=3773585)
```

#### Step 2: Converter para SPADL

```python
import socceraction.spadl as spadl

# Converter eventos StatsBomb → SPADL actions
actions = spadl.statsbomb.convert_to_actions(
    events,
    home_team_id=game.home_team_id,
    xy_fidelity_version=None,   # auto-detect
    shot_fidelity_version=None,  # auto-detect
)
```

**Schema SPADL** (12 atributos por ação):

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `game_id` | int | ID da partida |
| `original_event_id` | str | ID original do evento |
| `period_id` | int | Período (1, 2, ...) |
| `time_seconds` | float | Tempo em segundos desde início do período |
| `team_id` | int | ID do time |
| `player_id` | int | ID do jogador |
| `start_x` | float | Coordenada x inicial (0-105) |
| `start_y` | float | Coordenada y inicial (0-68) |
| `end_x` | float | Coordenada x final |
| `end_y` | float | Coordenada y final |
| `type_id` | int | Tipo de ação (0-21) |
| `result_id` | int | Resultado (0=fail, 1=success) |
| `bodypart_id` | int | Parte do corpo |
| `action_id` | int | Sequencial |

**Nota:** SPADL usa campo 105×68 (não 120×80 do StatsBomb). A conversão é automática.

**22 tipos de ação SPADL:** pass, cross, throw-in, crossed free kick, short free kick, crossed corner, short corner, take-on, foul, tackle, interception, shot, penalty shot, free kick shot, keeper save, keeper claim, keeper punch, keeper pick-up, clearance, bad touch, dribble, goal kick.

#### Step 3a: Calcular xT

```python
from socceraction.xthreat import ExpectedThreat

# Construir modelo xT
xT_model = ExpectedThreat(l=16, w=12, eps=1e-5)

# Fit: precisa de TODAS as ações da temporada/dataset (para calcular transições)
all_actions = pd.concat([actions_game1, actions_game2, ...])
xT_model.fit(all_actions)

# Avaliar ações de um jogo
xt_values = xT_model.rate(actions, use_interpolation=False)
# Returns: np.ndarray com xT delta para cada ação
# NaN para ações que não movem a bola (fouls, tackles, etc.)
```

**Parâmetros ExpectedThreat:**
- `l=16` — células no eixo x (default 16)
- `w=12` — células no eixo y (default 12)
- `eps=1e-5` — precisão de convergência

**Métodos:**
- `fit(actions: DataFrame[SPADLSchema]) -> ExpectedThreat` — treina o modelo
- `rate(actions: DataFrame[SPADLSchema], use_interpolation=False) -> np.ndarray` — retorna xT por ação
- `interpolator(kind="linear")` — retorna interpolador scipy para coordenadas contínuas

**Funções helper:**
- `scoring_prob(actions, l, w)` — P(gol) por zona
- `action_prob(actions, l, w)` — P(chutar) e P(mover) por zona
- `move_transition_matrix(actions, l, w)` — matriz de transição entre zonas
- `get_successful_move_actions(actions)` — filtra ações que movem bola com sucesso

#### Step 3b: Calcular VAEP

```python
from socceraction.vaep import VAEP

# Construir modelo VAEP
vaep_model = VAEP(
    xfns=None,           # usa features default se None
    nb_prev_actions=3,   # últimas 3 ações como contexto
)

# Fit: precisa de ações + labels (scores/concedes nos próximos N eventos)
# compute_features e compute_labels são chamados internamente
vaep_model.fit(actions, games)

# Avaliar
vaep_values = vaep_model.rate(actions, game_id=game_id)
# Returns: DataFrame com colunas: offensive_value, defensive_value, vaep_value
```

**Parâmetros VAEP:**
- `xfns` — lista de transformadores de features (None = default)
- `nb_prev_actions=3` — número de ações anteriores como contexto

**Métodos:**
- `fit(actions, games)` — treina gradient boosted classifier
- `rate(actions, game_id)` — retorna valores VAEP por ação
- `score(actions, game_id)` — retorna score do modelo
- `compute_features(actions)` — extrai features das game states
- `compute_labels(actions)` — computa labels (score/concede)

**Modelo interno:** Gradient boosted binary classifier (XGBoost ou CatBoost).

---

## 2. StatsBomb Event Schema — Campos Relevantes

### Pressure Events

Tipo: `"Pressure"`

Campos disponíveis via `sb.events()` (statsbombpy flattened):
- `type` = "Pressure"
- `player`, `player_id`, `team`
- `location` = [x, y]
- `duration` — duração da pressão em segundos
- `counterpress` — boolean, True se é counter-pressing (≤5s após turnover)
- `minute`, `second`, `timestamp`
- `period`
- `related_events` — IDs de eventos relacionados

### under_pressure (atributo em outros eventos)

**Campo:** `under_pressure` (boolean)

Presente em qualquer evento on-the-ball que ocorre durante uma pressão:
- Passes, Carries, Shots, Dribbles, Ball Receipt, etc.
- Calculado automaticamente: se timestamp do evento está dentro do intervalo de um Pressure event
- Duelos e dribbles **sempre** recebem o atributo se aplicável
- Carries podem ser "under pressure" por defensive events (não só Pressure)

**No statsbombpy:** coluna `under_pressure` no DataFrame de events.

### counterpress (atributo)

**Campo:** `counterpress` (boolean)

Aparece nos seguintes tipos de evento:
- Pressure
- Dribbled Past
- 50/50
- Duel
- Block
- Interception
- Foul Committed (não ofensiva)

**Definição:** ação defensiva dentro de 5 segundos de um turnover em jogo aberto.

### Ball Recovery Events

Tipo: `"Ball Recovery"`

Campos:
- `ball_recovery_recovery_failure` — boolean, True se a recuperação falhou
- `location`, `player`, `team`
- `under_pressure` — se a recuperação foi feita sob pressão

### Carry Events

Tipo: `"Carry"`

Campos:
- `location` — [x, y] início da condução
- `carry_end_location` — [x, y] fim da condução
- `under_pressure` — boolean
- `duration` — tempo de condução

### Duel Events

Tipo: `"Duel"`

Campos:
- `duel_type` — "Tackle" ou "Aerial Lost" etc.
- `duel_outcome` — "Won", "Lost", "Success", "Success In Play", "Success Out"
- `under_pressure`

---

## 3. StatsBomb Position Data

### Lineup JSON Schema

Cada jogador no lineup tem um array `positions`:

```json
{
  "player_id": 5503,
  "player_name": "Lionel Messi",
  "positions": [
    {
      "position_id": 17,
      "position": "Right Wing",
      "from": "00:00",
      "to": "45:00",
      "from_period": 1,
      "to_period": 1,
      "start_reason": "Starting XI",
      "end_reason": "Half Time"
    }
  ]
}
```

### Mapeamento Completo position_id → Posição → Grupo

| ID | Posição | Abreviação | Grupo |
|----|---------|------------|-------|
| 1 | Goalkeeper | GK | GK |
| 2 | Right Back | RB | DEF |
| 3 | Right Center Back | RCB | DEF |
| 4 | Center Back | CB | DEF |
| 5 | Left Center Back | LCB | DEF |
| 6 | Left Back | LB | DEF |
| 7 | Right Wing Back | RWB | DEF |
| 8 | Left Wing Back | LWB | DEF |
| 9 | Right Defensive Midfield | RDM | MID |
| 10 | Center Defensive Midfield | CDM | MID |
| 11 | Left Defensive Midfield | LDM | MID |
| 12 | Right Midfield | RM | MID |
| 13 | Right Center Midfield | RCM | MID |
| 14 | Center Midfield | CM | MID |
| 15 | Left Center Midfield | LCM | MID |
| 16 | Left Midfield | LM | MID |
| 17 | Right Wing | RW | FWD |
| 18 | Right Attacking Midfield | RAM | MID |
| 19 | Center Attacking Midfield | CAM | MID |
| 20 | Left Attacking Midfield | LAM | MID |
| 21 | Left Wing | LW | FWD |
| 22 | Right Center Forward | RCF | FWD |
| 23 | Striker | ST | FWD |
| 24 | Left Center Forward | LCF | FWD |
| 25 | Second Striker | SS | FWD |

**Nota:** Wingers (17, 21) classificados como FWD. Wing-backs (7, 8) como DEF. Attacking mids (18-20) como MID.

### Acesso via statsbombpy

```python
lineups = sb.lineups(match_id=3773585)
# Returns: dict[team_name] -> DataFrame com player_id, player_name, jersey_number, positions
```

O campo `positions` é uma lista de dicts. Para obter posição primária: pegar a primeira posição (start_reason="Starting XI").

---

## 4. Campos para Pressing Metrics

### PPDA (Passes Per Defensive Action)

Calculável com:
- **Passes do adversário:** `events[(events.type == "Pass") & (events.team == adversário)]`
- **Ações defensivas:** `events[events.type.isin(["Pressure", "Tackle", "Interception", "Foul Committed", "Block"]) & (events.team == time)]`
- **Filtro por zona:** usar `location[0]` para filtrar por terço do campo

### Counter-pressing Fraction

- Identificar turnovers: momentos onde posse muda de time
- Filtrar Pressure events com `counterpress == True` nos ≤5s seguintes
- `counter_pressing_fraction = turnover_com_pressão / total_turnovers`

### High Turnovers

- Ball Recovery events onde `location[0] >= 80` (≤40m do gol adversário, em coordenadas StatsBomb 120x80)
- Filtrar por time

### Pressing Success Rate

- Pressure events → verificar se nos próximos N eventos (≤5s) há Ball Recovery do mesmo time
- `success_rate = pressões_com_recuperação / total_pressões`

---

## Implications for Football Moneyball

### Decisões de implementação

1. **xT: implementação custom vs. socceraction**
   - socceraction tem xT pronto (`ExpectedThreat` class), mas usa coordenadas SPADL (105×68)
   - Opção: usar socceraction diretamente (converte StatsBomb→SPADL automaticamente)
   - Opção: implementar custom com coordenadas StatsBomb nativas (120×80)
   - **Recomendação:** usar socceraction — menos código, já testado, MIT license

2. **VAEP: depende de socceraction**
   - Não há como implementar VAEP sem a lib — o modelo ML precisa do framework SPADL+features
   - **Recomendação:** usar socceraction para VAEP

3. **Pressing: implementação custom**
   - Não existe lib pronta — precisa extrair dos events StatsBomb diretamente
   - Todos os campos necessários estão disponíveis (pressure, counterpress, under_pressure, ball_recovery, location)

4. **Posições: dados disponíveis**
   - StatsBomb fornece position_id nos lineups
   - Mapeamento para 4 grupos (GK/DEF/MID/FWD) é direto
   - Acesso via `sb.lineups()` — precisa parsear o array `positions`

5. **Coordenadas: cuidado com conversão**
   - StatsBomb: 120×80 (gol em x=120)
   - SPADL/socceraction: 105×68
   - Nosso código atual usa 120×80 (StatsBomb nativo)
   - Se usar socceraction para xT/VAEP, os valores são calculados em 105×68 mas isso não afeta o resultado — o modelo treina e avalia nas mesmas coordenadas

---

## Sources

- [socceraction GitHub](https://github.com/ML-KULeuven/socceraction)
- [socceraction PyPI](https://pypi.org/project/socceraction/)
- [socceraction ReadTheDocs](https://socceraction.readthedocs.io/)
- [ExpectedThreat API](https://socceraction.readthedocs.io/en/latest/api/generated/socceraction.xthreat.ExpectedThreat.html)
- [VAEP API](https://socceraction.readthedocs.io/en/latest/api/generated/socceraction.vaep.VAEP.html)
- [SPADL format docs](https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html)
- [StatsBomb SPADL converter source](https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/spadl/statsbomb.py)
- [StatsBomb notebook — load & convert](https://github.com/ML-KULeuven/socceraction/blob/master/public-notebooks/1-load-and-convert-statsbomb-data.ipynb)
- [StatsBomb open-data GitHub](https://github.com/statsbomb/open-data)
- [StatsBomb Open Data Events v4.0.0](https://www.scribd.com/document/816093666/Open-Data-Events-v4-0-0)
- [StatsBomb lineup example](https://github.com/statsbomb/open-data/blob/master/data/lineups/7298.json)
- [StatsBomb counter-pressing article](https://blogarchive.statsbomb.com/articles/soccer/how-statsbomb-data-helps-measure-counter-pressing/)
- [statsbombpy GitHub](https://github.com/statsbomb/statsbombpy)
- [Working with StatsBomb Data — Trym Sorum](https://medium.com/@trym.sorum/a-guide-to-statsbomb-data-in-python-7fa0443cbc41)
