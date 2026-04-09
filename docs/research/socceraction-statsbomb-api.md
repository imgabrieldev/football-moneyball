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
> Sources: [listed at the end]

## Context

To implement the [[state-of-the-art-engine]] pitch, we need to understand exactly how to use the `socceraction` library for xT/VAEP and which StatsBomb fields are available for pressing metrics and position-aware embeddings.

---

## 1. socceraction — Full API

### Installation

```bash
pip install socceraction>=1.5
```

MIT-licensed library, v1.5.3 (Dec 2024). **Not under active development** — used for reproducibility of research. Supports StatsBomb, Opta, Wyscout, Stats Perform, WhoScored.

### Pipeline: StatsBomb → SPADL → xT/VAEP

#### Step 1: Load StatsBomb data

```python
from socceraction.data.statsbomb import StatsBombLoader

# Open data (no credentials)
SBL = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

# List
competitions = SBL.competitions()
games = SBL.games(competition_id=11, season_id=90)

# Load events of a match
events = SBL.events(game_id=3773585)

# Load lineup/players
players = SBL.players(game_id=3773585)
teams = SBL.teams(game_id=3773585)
```

#### Step 2: Convert to SPADL

```python
import socceraction.spadl as spadl

# Convert StatsBomb events → SPADL actions
actions = spadl.statsbomb.convert_to_actions(
    events,
    home_team_id=game.home_team_id,
    xy_fidelity_version=None,   # auto-detect
    shot_fidelity_version=None,  # auto-detect
)
```

**SPADL schema** (12 attributes per action):

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | int | Match ID |
| `original_event_id` | str | Original event ID |
| `period_id` | int | Period (1, 2, ...) |
| `time_seconds` | float | Time in seconds since start of the period |
| `team_id` | int | Team ID |
| `player_id` | int | Player ID |
| `start_x` | float | Starting x coordinate (0-105) |
| `start_y` | float | Starting y coordinate (0-68) |
| `end_x` | float | Ending x coordinate |
| `end_y` | float | Ending y coordinate |
| `type_id` | int | Action type (0-21) |
| `result_id` | int | Result (0=fail, 1=success) |
| `bodypart_id` | int | Body part |
| `action_id` | int | Sequence number |

**Note:** SPADL uses a 105×68 pitch (not the 120×80 of StatsBomb). The conversion is automatic.

**22 SPADL action types:** pass, cross, throw-in, crossed free kick, short free kick, crossed corner, short corner, take-on, foul, tackle, interception, shot, penalty shot, free kick shot, keeper save, keeper claim, keeper punch, keeper pick-up, clearance, bad touch, dribble, goal kick.

#### Step 3a: Compute xT

```python
from socceraction.xthreat import ExpectedThreat

# Build the xT model
xT_model = ExpectedThreat(l=16, w=12, eps=1e-5)

# Fit: requires ALL actions from the season/dataset (to compute transitions)
all_actions = pd.concat([actions_game1, actions_game2, ...])
xT_model.fit(all_actions)

# Evaluate actions of a match
xt_values = xT_model.rate(actions, use_interpolation=False)
# Returns: np.ndarray with xT delta for each action
# NaN for actions that don't move the ball (fouls, tackles, etc.)
```

**ExpectedThreat parameters:**

- `l=16` — cells on the x axis (default 16)
- `w=12` — cells on the y axis (default 12)
- `eps=1e-5` — convergence precision

**Methods:**

- `fit(actions: DataFrame[SPADLSchema]) -> ExpectedThreat` — trains the model
- `rate(actions: DataFrame[SPADLSchema], use_interpolation=False) -> np.ndarray` — returns xT per action
- `interpolator(kind="linear")` — returns a scipy interpolator for continuous coordinates

**Helper functions:**

- `scoring_prob(actions, l, w)` — P(goal) per zone
- `action_prob(actions, l, w)` — P(shoot) and P(move) per zone
- `move_transition_matrix(actions, l, w)` — transition matrix between zones
- `get_successful_move_actions(actions)` — filters actions that successfully move the ball

#### Step 3b: Compute VAEP

```python
from socceraction.vaep import VAEP

# Build VAEP model
vaep_model = VAEP(
    xfns=None,           # uses default features if None
    nb_prev_actions=3,   # last 3 actions as context
)

# Fit: needs actions + labels (scores/concedes in the next N events)
# compute_features and compute_labels are called internally
vaep_model.fit(actions, games)

# Evaluate
vaep_values = vaep_model.rate(actions, game_id=game_id)
# Returns: DataFrame with columns: offensive_value, defensive_value, vaep_value
```

**VAEP parameters:**

- `xfns` — list of feature transformers (None = default)
- `nb_prev_actions=3` — number of previous actions as context

**Methods:**

- `fit(actions, games)` — trains gradient boosted classifier
- `rate(actions, game_id)` — returns VAEP values per action
- `score(actions, game_id)` — returns model score
- `compute_features(actions)` — extracts features from game states
- `compute_labels(actions)` — computes labels (score/concede)

**Internal model:** Gradient boosted binary classifier (XGBoost or CatBoost).

---

## 2. StatsBomb Event Schema — Relevant Fields

### Pressure Events

Type: `"Pressure"`

Fields available via `sb.events()` (statsbombpy flattened):

- `type` = "Pressure"
- `player`, `player_id`, `team`
- `location` = [x, y]
- `duration` — pressure duration in seconds
- `counterpress` — boolean, True if it is counter-pressing (≤5s after turnover)
- `minute`, `second`, `timestamp`
- `period`
- `related_events` — IDs of related events

### under_pressure (attribute on other events)

**Field:** `under_pressure` (boolean)

Present on any on-the-ball event that happens during a pressure:

- Passes, Carries, Shots, Dribbles, Ball Receipt, etc.
- Computed automatically: if the event's timestamp falls within a Pressure event's interval
- Duels and dribbles **always** receive the attribute when applicable
- Carries may be "under pressure" from defensive events (not just Pressure)

**In statsbombpy:** `under_pressure` column on the events DataFrame.

### counterpress (attribute)

**Field:** `counterpress` (boolean)

Appears on the following event types:

- Pressure
- Dribbled Past
- 50/50
- Duel
- Block
- Interception
- Foul Committed (non-offensive)

**Definition:** defensive action within 5 seconds of a turnover in open play.

### Ball Recovery Events

Type: `"Ball Recovery"`

Fields:

- `ball_recovery_recovery_failure` — boolean, True if the recovery failed
- `location`, `player`, `team`
- `under_pressure` — if the recovery was made under pressure

### Carry Events

Type: `"Carry"`

Fields:

- `location` — [x, y] start of the carry
- `carry_end_location` — [x, y] end of the carry
- `under_pressure` — boolean
- `duration` — carry time

### Duel Events

Type: `"Duel"`

Fields:

- `duel_type` — "Tackle" or "Aerial Lost" etc.
- `duel_outcome` — "Won", "Lost", "Success", "Success In Play", "Success Out"
- `under_pressure`

---

## 3. StatsBomb Position Data

### Lineup JSON Schema

Each player in the lineup has a `positions` array:

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

### Complete position_id → Position → Group Mapping

| ID | Position | Abbreviation | Group |
|----|----------|--------------|-------|
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

**Note:** Wingers (17, 21) classified as FWD. Wing-backs (7, 8) as DEF. Attacking mids (18-20) as MID.

### Access via statsbombpy

```python
lineups = sb.lineups(match_id=3773585)
# Returns: dict[team_name] -> DataFrame with player_id, player_name, jersey_number, positions
```

The `positions` field is a list of dicts. To obtain the primary position: take the first position (start_reason="Starting XI").

---

## 4. Fields for Pressing Metrics

### PPDA (Passes Per Defensive Action)

Computable with:

- **Opponent passes:** `events[(events.type == "Pass") & (events.team == opponent)]`
- **Defensive actions:** `events[events.type.isin(["Pressure", "Tackle", "Interception", "Foul Committed", "Block"]) & (events.team == team)]`
- **Zone filter:** use `location[0]` to filter by pitch third

### Counter-pressing Fraction

- Identify turnovers: moments where possession changes teams
- Filter Pressure events with `counterpress == True` in the next ≤5s
- `counter_pressing_fraction = turnovers_with_pressure / total_turnovers`

### High Turnovers

- Ball Recovery events where `location[0] >= 80` (≤40m from the opponent's goal, in StatsBomb 120x80 coordinates)
- Filter by team

### Pressing Success Rate

- Pressure events → check if in the next N events (≤5s) there is a Ball Recovery by the same team
- `success_rate = pressures_with_recovery / total_pressures`

---

## Implications for Football Moneyball

### Implementation decisions

1. **xT: custom implementation vs. socceraction**
   - socceraction has xT ready (`ExpectedThreat` class), but uses SPADL coordinates (105×68)
   - Option: use socceraction directly (converts StatsBomb→SPADL automatically)
   - Option: implement custom with native StatsBomb coordinates (120×80)
   - **Recommendation:** use socceraction — less code, already tested, MIT license

2. **VAEP: depends on socceraction**
   - There is no way to implement VAEP without the library — the ML model needs the SPADL+features framework
   - **Recommendation:** use socceraction for VAEP

3. **Pressing: custom implementation**
   - No ready-made library exists — need to extract directly from StatsBomb events
   - All the required fields are available (pressure, counterpress, under_pressure, ball_recovery, location)

4. **Positions: data available**
   - StatsBomb provides position_id in the lineups
   - Mapping to 4 groups (GK/DEF/MID/FWD) is straightforward
   - Access via `sb.lineups()` — need to parse the `positions` array

5. **Coordinates: beware of conversion**
   - StatsBomb: 120×80 (goal at x=120)
   - SPADL/socceraction: 105×68
   - Our current code uses 120×80 (native StatsBomb)
   - If using socceraction for xT/VAEP, the values are computed in 105×68 but this doesn't affect the result — the model trains and evaluates on the same coordinates

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
