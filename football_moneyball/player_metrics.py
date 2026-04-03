"""Modulo de extracao de metricas individuais de jogadores a partir de dados StatsBomb.

Extrai aproximadamente 30 metricas por jogador de cada partida, incluindo
estatisticas ofensivas, defensivas, de posse e de passe.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsbombpy import sb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Retorna a coluna se existir, caso contrario retorna Series de NaN."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def _count_events(events: pd.DataFrame, mask: pd.Series, player_col: str = "player") -> pd.Series:
    """Conta eventos agrupados por jogador aplicando uma mascara booleana."""
    return events.loc[mask].groupby(player_col).size()


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_match_metrics(match_id: int) -> pd.DataFrame:
    """Extrai metricas individuais de cada jogador para uma partida.

    Busca os eventos da partida via ``statsbombpy`` e calcula ~30 metricas
    por jogador, retornando um DataFrame compativel com o schema do banco.

    Parameters
    ----------
    match_id : int
        Identificador da partida no StatsBomb.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: ``player_id``, ``player_name``, ``team`` e
        todas as metricas calculadas.
    """
    events = sb.events(match_id=match_id)

    if events.empty:
        warnings.warn(f"Nenhum evento encontrado para match_id={match_id}")
        return pd.DataFrame()

    # Ensure we have the columns we need; statsbombpy flattens nested fields
    # with underscores (e.g. shot_statsbomb_xg, pass_goal_assist, etc.)

    # ------------------------------------------------------------------
    # Build player roster from events
    # ------------------------------------------------------------------
    player_info = (
        events[events["player"].notna()]
        .groupby("player")
        .agg(
            player_id=("player_id", "first"),
            team=("team", "first"),
        )
        .reset_index()
        .rename(columns={"player": "player_name"})
    )

    players = player_info["player_name"]
    metrics: dict[str, dict[str, Any]] = {p: {} for p in players}

    # ------------------------------------------------------------------
    # Shots & Goals
    # ------------------------------------------------------------------
    shots_mask = events["type"] == "Shot"
    shots = events[shots_mask].copy()

    for player_name, grp in shots.groupby("player"):
        metrics[player_name]["shots"] = len(grp)
        metrics[player_name]["goals"] = int(
            (grp["shot_outcome"].isin(["Goal"])).sum()
            if "shot_outcome" in grp.columns
            else 0
        )
        metrics[player_name]["shots_on_target"] = int(
            grp["shot_outcome"].isin(["Goal", "Saved", "Saved To Post"]).sum()
            if "shot_outcome" in grp.columns
            else 0
        )
        metrics[player_name]["xg"] = float(
            _safe_col(grp, "shot_statsbomb_xg").sum()
        )

    # ------------------------------------------------------------------
    # Passes, assists, xA
    # ------------------------------------------------------------------
    passes_mask = events["type"] == "Pass"
    passes = events[passes_mask].copy()

    # Build xA lookup: for each pass flagged as goal_assist, find the
    # corresponding shot's xG.  The shot is the next event by the
    # recipient in the same possession.
    xa_by_player: dict[str, float] = {}
    if "pass_goal_assist" in passes.columns:
        assist_passes = passes[passes["pass_goal_assist"] == True]  # noqa: E712
        for _, ap in assist_passes.iterrows():
            passer = ap["player"]
            # Find the goal shot in the same possession
            possession_id = ap.get("possession", None)
            if possession_id is not None:
                poss_shots = shots[
                    (shots["possession"] == possession_id)
                    & (shots.get("shot_outcome", pd.Series()) == "Goal")
                    if "shot_outcome" in shots.columns
                    else shots["possession"] == possession_id
                ]
                if not poss_shots.empty and "shot_statsbomb_xg" in poss_shots.columns:
                    xa_val = poss_shots["shot_statsbomb_xg"].iloc[0]
                else:
                    xa_val = 0.0
            else:
                xa_val = 0.0
            xa_by_player[passer] = xa_by_player.get(passer, 0.0) + float(
                xa_val if not pd.isna(xa_val) else 0.0
            )

    for player_name, grp in passes.groupby("player"):
        total = len(grp)
        metrics[player_name]["passes"] = total

        # Successful passes: outcome is NaN (successful) or 'Complete'
        if "pass_outcome" in grp.columns:
            completed = grp["pass_outcome"].isna() | (grp["pass_outcome"] == "Complete")
            metrics[player_name]["passes_completed"] = int(completed.sum())
        else:
            # If column missing, assume all successful
            metrics[player_name]["passes_completed"] = total

        metrics[player_name]["pass_pct"] = round(
            metrics[player_name]["passes_completed"] / total * 100, 1
        ) if total > 0 else 0.0

        # Assists
        if "pass_goal_assist" in grp.columns:
            metrics[player_name]["assists"] = int(
                grp["pass_goal_assist"].fillna(False).astype(bool).sum()
            )
        else:
            metrics[player_name]["assists"] = 0

        # xA
        metrics[player_name]["xa"] = round(xa_by_player.get(player_name, 0.0), 2)

        # Key passes (shot assists)
        if "pass_shot_assist" in grp.columns:
            metrics[player_name]["key_passes"] = int(
                grp["pass_shot_assist"].fillna(False).astype(bool).sum()
            )
        else:
            metrics[player_name]["key_passes"] = 0

        # Through balls
        if "pass_technique" in grp.columns:
            metrics[player_name]["through_balls"] = int(
                (grp["pass_technique"] == "Through Ball").sum()
            )
        else:
            metrics[player_name]["through_balls"] = 0

        # Crosses
        if "pass_cross" in grp.columns:
            metrics[player_name]["crosses"] = int(
                grp["pass_cross"].fillna(False).astype(bool).sum()
            )
        else:
            metrics[player_name]["crosses"] = 0

        # Progressive passes: move ball >= 10 yards closer to goal
        if all(
            c in grp.columns
            for c in ["location", "pass_end_location"]
        ):
            prog_count = 0
            for _, row in grp.iterrows():
                try:
                    start = row["location"]
                    end = row["pass_end_location"]
                    if isinstance(start, list) and isinstance(end, list):
                        # StatsBomb pitch is 120x80; goal at x=120
                        start_dist = 120 - start[0]
                        end_dist = 120 - end[0]
                        # 10 yards closer to opponent goal
                        if start_dist - end_dist >= 10:
                            prog_count += 1
                except (TypeError, IndexError):
                    continue
            metrics[player_name]["progressive_passes"] = prog_count
        else:
            metrics[player_name]["progressive_passes"] = 0

    # ------------------------------------------------------------------
    # Carries & progressive carries
    # ------------------------------------------------------------------
    carries_mask = events["type"] == "Carry"
    carries = events[carries_mask].copy()

    for player_name, grp in carries.groupby("player"):
        metrics[player_name]["carries"] = len(grp)

        if all(c in grp.columns for c in ["location", "carry_end_location"]):
            prog_count = 0
            for _, row in grp.iterrows():
                try:
                    start = row["location"]
                    end = row["carry_end_location"]
                    if isinstance(start, list) and isinstance(end, list):
                        start_dist = 120 - start[0]
                        end_dist = 120 - end[0]
                        if start_dist - end_dist >= 10:
                            prog_count += 1
                except (TypeError, IndexError):
                    continue
            metrics[player_name]["progressive_carries"] = prog_count
        else:
            metrics[player_name]["progressive_carries"] = 0

    # ------------------------------------------------------------------
    # Defensive actions
    # ------------------------------------------------------------------
    for event_type, metric_name in [
        ("Duel", None),  # handled separately
        ("Interception", "interceptions"),
        ("Block", "blocks"),
        ("Clearance", "clearances"),
    ]:
        if metric_name is None:
            continue
        mask = events["type"] == event_type
        for player_name, count in _count_events(events, mask).items():
            metrics.setdefault(player_name, {})[metric_name] = int(count)

    # Tackles (subtype of Duel or standalone type depending on version)
    tackle_mask = events["type"] == "Duel"
    if "duel_type" in events.columns:
        tackle_mask = tackle_mask & (events["duel_type"] == "Tackle")
    for player_name, count in _count_events(events, tackle_mask).items():
        metrics.setdefault(player_name, {})["tackles"] = int(count)

    # Aerials won / lost
    aerial_mask = events["type"] == "Duel"
    if "duel_type" in events.columns:
        aerial_mask = aerial_mask & (
            events["duel_type"].str.contains("Aerial", case=False, na=False)
        )
    aerials = events[aerial_mask].copy()
    for player_name, grp in aerials.groupby("player"):
        if "duel_outcome" in grp.columns:
            won = grp["duel_outcome"].isin(["Won", "Success", "Success In Play", "Success Out"]).sum()
            lost = len(grp) - won
        else:
            won = 0
            lost = len(grp)
        metrics.setdefault(player_name, {})["aerials_won"] = int(won)
        metrics.setdefault(player_name, {})["aerials_lost"] = int(lost)

    # ------------------------------------------------------------------
    # Fouls
    # ------------------------------------------------------------------
    foul_mask = events["type"] == "Foul Committed"
    for player_name, count in _count_events(events, foul_mask).items():
        metrics.setdefault(player_name, {})["fouls_committed"] = int(count)

    foul_won_mask = events["type"] == "Foul Won"
    for player_name, count in _count_events(events, foul_won_mask).items():
        metrics.setdefault(player_name, {})["fouls_won"] = int(count)

    # ------------------------------------------------------------------
    # Dribbles
    # ------------------------------------------------------------------
    dribble_mask = events["type"] == "Dribble"
    dribbles = events[dribble_mask].copy()
    for player_name, grp in dribbles.groupby("player"):
        metrics.setdefault(player_name, {})["dribbles_attempted"] = len(grp)
        if "dribble_outcome" in grp.columns:
            metrics[player_name]["dribbles_completed"] = int(
                (grp["dribble_outcome"] == "Complete").sum()
            )
        else:
            metrics[player_name]["dribbles_completed"] = 0

    # ------------------------------------------------------------------
    # Touches (total events attributed to each player)
    # ------------------------------------------------------------------
    player_events = events[events["player"].notna()]
    for player_name, count in player_events.groupby("player").size().items():
        metrics.setdefault(player_name, {})["touches"] = int(count)

    # ------------------------------------------------------------------
    # Dispossessed / Miscontrol
    # ------------------------------------------------------------------
    dispossessed_mask = events["type"].isin(["Dispossessed", "Miscontrol"])
    for player_name, count in _count_events(events, dispossessed_mask).items():
        metrics.setdefault(player_name, {})["dispossessed"] = int(count)

    # ------------------------------------------------------------------
    # Pressures
    # ------------------------------------------------------------------
    pressure_mask = events["type"] == "Pressure"
    for player_name, count in _count_events(events, pressure_mask).items():
        metrics.setdefault(player_name, {})["pressures"] = int(count)

    # Pressure regains (counterpressure)
    if "counterpress" in events.columns:
        cp_mask = pressure_mask & (events["counterpress"] == True)  # noqa: E712
    else:
        cp_mask = pd.Series(False, index=events.index)
    for player_name, count in _count_events(events, cp_mask).items():
        metrics.setdefault(player_name, {})["pressure_regains"] = int(count)

    # ------------------------------------------------------------------
    # v0.2.0 — Progressive Receptions
    # ------------------------------------------------------------------
    receipt_mask = events["type"] == "Ball Receipt*"
    if receipt_mask.any() and "location" in events.columns:
        # Find the preceding pass for each receipt to check progressiveness
        for player_name, grp in events[receipt_mask].groupby("player"):
            prog_rec = 0
            for _, row in grp.iterrows():
                try:
                    rec_loc = row["location"]
                    if not isinstance(rec_loc, list) or len(rec_loc) < 2:
                        continue
                    # Find the pass that led to this receipt (same possession)
                    poss = row.get("possession", None)
                    if poss is None:
                        continue
                    poss_passes = passes[
                        (passes["possession"] == poss)
                        & (passes.index < row.name)
                    ]
                    if poss_passes.empty:
                        continue
                    last_pass = poss_passes.iloc[-1]
                    pass_loc = last_pass.get("location")
                    if isinstance(pass_loc, list) and len(pass_loc) >= 1:
                        start_dist = 120 - pass_loc[0]
                        end_dist = 120 - rec_loc[0]
                        if start_dist - end_dist >= 10:
                            prog_rec += 1
                except (TypeError, IndexError, KeyError):
                    continue
            metrics.setdefault(player_name, {})["progressive_receptions"] = prog_rec

    # ------------------------------------------------------------------
    # v0.2.0 — Shot Quality (Big Chances)
    # ------------------------------------------------------------------
    for player_name, grp in shots.groupby("player"):
        if "shot_statsbomb_xg" in grp.columns:
            big = grp["shot_statsbomb_xg"].fillna(0) >= 0.3
            metrics.setdefault(player_name, {})["big_chances"] = int(big.sum())
            if "shot_outcome" in grp.columns:
                big_missed = big & ~grp["shot_outcome"].isin(["Goal"])
                metrics[player_name]["big_chances_missed"] = int(big_missed.sum())
            else:
                metrics[player_name]["big_chances_missed"] = int(big.sum())
        else:
            metrics.setdefault(player_name, {})["big_chances"] = 0
            metrics[player_name]["big_chances_missed"] = 0

    # ------------------------------------------------------------------
    # v0.2.0 — Pass Breakdown (short/medium/long) + under pressure + switches
    # ------------------------------------------------------------------
    for player_name, grp in passes.groupby("player"):
        p_short = p_short_c = p_med = p_med_c = p_long = p_long_c = 0
        p_under = p_under_c = 0
        switches = 0

        has_locs = all(c in grp.columns for c in ["location", "pass_end_location"])
        has_outcome = "pass_outcome" in grp.columns

        for _, row in grp.iterrows():
            try:
                start = row["location"] if has_locs else None
                end = row["pass_end_location"] if has_locs else None

                if isinstance(start, list) and isinstance(end, list):
                    dist = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    is_complete = (
                        pd.isna(row["pass_outcome"]) or row["pass_outcome"] == "Complete"
                    ) if has_outcome else True

                    # Short / medium / long
                    if dist < 15:
                        p_short += 1
                        p_short_c += int(is_complete)
                    elif dist < 30:
                        p_med += 1
                        p_med_c += int(is_complete)
                    else:
                        p_long += 1
                        p_long_c += int(is_complete)

                    # Switches of play (lateral > 30 yards)
                    if abs(end[1] - start[1]) > 30:
                        switches += 1

                # Under pressure
                if row.get("under_pressure") is True:
                    p_under += 1
                    if has_outcome:
                        is_c = pd.isna(row["pass_outcome"]) or row["pass_outcome"] == "Complete"
                        p_under_c += int(is_c)
            except (TypeError, IndexError, KeyError):
                continue

        m = metrics.setdefault(player_name, {})
        m["passes_short"] = p_short
        m["passes_short_completed"] = p_short_c
        m["passes_medium"] = p_med
        m["passes_medium_completed"] = p_med_c
        m["passes_long"] = p_long
        m["passes_long_completed"] = p_long_c
        m["passes_under_pressure"] = p_under
        m["passes_under_pressure_completed"] = p_under_c
        m["switches_of_play"] = switches

    # ------------------------------------------------------------------
    # v0.2.0 — Ground Duels & Tackle Success Rate
    # ------------------------------------------------------------------
    ground_duel_mask = events["type"] == "Duel"
    if "duel_type" in events.columns:
        ground_duel_mask = ground_duel_mask & (events["duel_type"] == "Tackle")
    ground_duels = events[ground_duel_mask].copy()
    _won_outcomes = {"Won", "Success", "Success In Play", "Success Out"}
    for player_name, grp in ground_duels.groupby("player"):
        total_gd = len(grp)
        won_gd = 0
        if "duel_outcome" in grp.columns:
            won_gd = int(grp["duel_outcome"].isin(_won_outcomes).sum())
        m = metrics.setdefault(player_name, {})
        m["ground_duels_won"] = won_gd
        m["ground_duels_total"] = total_gd
        m["tackle_success_rate"] = round(won_gd / total_gd * 100, 1) if total_gd > 0 else 0.0

    # ------------------------------------------------------------------
    # Minutes played (estimated from first/last event timestamp)
    # ------------------------------------------------------------------
    if "timestamp" in events.columns:
        for player_name, grp in player_events.groupby("player"):
            try:
                timestamps = pd.to_timedelta(grp["timestamp"])
                mins = (timestamps.max() - timestamps.min()).total_seconds() / 60
                metrics.setdefault(player_name, {})["minutes_played"] = round(mins, 1)
            except Exception:
                metrics.setdefault(player_name, {})["minutes_played"] = 0.0
    else:
        for player_name in metrics:
            metrics[player_name]["minutes_played"] = 0.0

    # ------------------------------------------------------------------
    # Assemble final DataFrame
    # ------------------------------------------------------------------
    all_metric_cols = [
        "goals", "assists", "shots", "shots_on_target", "xg", "xa",
        "passes", "passes_completed", "pass_pct",
        "progressive_passes", "progressive_carries", "progressive_receptions",
        "key_passes", "through_balls", "crosses",
        "tackles", "interceptions", "blocks", "clearances",
        "aerials_won", "aerials_lost",
        "fouls_committed", "fouls_won",
        "dribbles_attempted", "dribbles_completed",
        "touches", "carries", "dispossessed",
        "pressures", "pressure_regains",
        "big_chances", "big_chances_missed",
        "passes_short", "passes_short_completed",
        "passes_medium", "passes_medium_completed",
        "passes_long", "passes_long_completed",
        "passes_under_pressure", "passes_under_pressure_completed",
        "switches_of_play",
        "ground_duels_won", "ground_duels_total", "tackle_success_rate",
        "minutes_played",
    ]

    rows = []
    for _, pinfo in player_info.iterrows():
        name = pinfo["player_name"]
        row = {
            "player_id": pinfo["player_id"],
            "player_name": name,
            "team": pinfo["team"],
        }
        pm = metrics.get(name, {})
        for col in all_metric_cols:
            row[col] = pm.get(col, 0)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Position extraction
# ---------------------------------------------------------------------------

# StatsBomb position_id → group mapping
POSITION_GROUP_MAP: dict[int, str] = {
    1: "GK",
    2: "DEF", 3: "DEF", 4: "DEF", 5: "DEF", 6: "DEF", 7: "DEF", 8: "DEF",
    9: "MID", 10: "MID", 11: "MID", 12: "MID", 13: "MID", 14: "MID",
    15: "MID", 16: "MID",
    17: "FWD", 18: "MID", 19: "MID", 20: "MID", 21: "FWD",
    22: "FWD", 23: "FWD", 24: "FWD", 25: "FWD",
}


def extract_player_positions(match_id: int) -> dict[int, str]:
    """Extrai a posicao primaria de cada jogador a partir dos lineups.

    Usa o primeiro registro de posicao (start_reason='Starting XI') para
    determinar o grupo posicional (GK, DEF, MID, FWD).

    Parameters
    ----------
    match_id : int
        Identificador da partida no StatsBomb.

    Returns
    -------
    dict[int, str]
        Mapeamento player_id → position_group ('GK', 'DEF', 'MID', 'FWD').
    """
    lineups = sb.lineups(match_id=match_id)
    positions: dict[int, str] = {}

    for team_name, team_df in lineups.items():
        for _, player_row in team_df.iterrows():
            player_id = player_row.get("player_id")
            if player_id is None:
                continue

            pos_list = player_row.get("positions", [])
            if not isinstance(pos_list, list) or not pos_list:
                continue

            # Use first position (usually Starting XI)
            first_pos = pos_list[0]
            if isinstance(first_pos, dict):
                pos_id = first_pos.get("position_id")
                if pos_id is not None:
                    positions[int(player_id)] = POSITION_GROUP_MAP.get(
                        int(pos_id), "MID"
                    )

    return positions


# ---------------------------------------------------------------------------
# Match info
# ---------------------------------------------------------------------------

def extract_match_info(match_id: int) -> dict[str, Any]:
    """Retorna metadados de uma partida especifica.

    Busca as informacoes da partida (competicao, temporada, times, placar)
    a partir dos dados abertos do StatsBomb.

    Parameters
    ----------
    match_id : int
        Identificador da partida no StatsBomb.

    Returns
    -------
    dict
        Dicionario com chaves: ``match_id``, ``competition``, ``season``,
        ``match_date``, ``home_team``, ``away_team``, ``home_score``,
        ``away_score``.
    """
    comps = sb.competitions()
    for _, comp in comps.iterrows():
        try:
            matches = sb.matches(
                competition_id=comp["competition_id"],
                season_id=comp["season_id"],
            )
        except Exception:
            continue

        match_row = matches[matches["match_id"] == match_id]
        if not match_row.empty:
            m = match_row.iloc[0]
            return {
                "match_id": int(m["match_id"]),
                "competition": m.get("competition", comp.get("competition_name", "")),
                "season": m.get("season", comp.get("season_name", "")),
                "match_date": str(m.get("match_date", "")),
                "home_team": m.get("home_team", ""),
                "away_team": m.get("away_team", ""),
                "home_score": int(m.get("home_score", 0)),
                "away_score": int(m.get("away_score", 0)),
            }

    warnings.warn(f"Partida {match_id} nao encontrada nos dados abertos do StatsBomb.")
    return {"match_id": match_id}


# ---------------------------------------------------------------------------
# Competition helpers
# ---------------------------------------------------------------------------

def get_competition_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Retorna todas as partidas de uma competicao/temporada.

    Parameters
    ----------
    competition_id : int
        Identificador da competicao no StatsBomb.
    season_id : int
        Identificador da temporada no StatsBomb.

    Returns
    -------
    pd.DataFrame
        DataFrame com informacoes de todas as partidas encontradas.
    """
    return sb.matches(competition_id=competition_id, season_id=season_id)


def get_free_competitions() -> pd.DataFrame:
    """Retorna as competicoes disponiveis nos dados abertos do StatsBomb.

    Filtra apenas competicoes marcadas como dados abertos (open data).

    Returns
    -------
    pd.DataFrame
        DataFrame com as competicoes disponiveis gratuitamente.
    """
    comps = sb.competitions()
    # Some versions of statsbombpy do not expose a dedicated open-data flag.
    # When the column is absent we return everything (all sb.competitions()
    # results from the free tier are open data by default).
    if "match_available" in comps.columns:
        return comps[comps["match_available"].notna()].reset_index(drop=True)
    return comps
