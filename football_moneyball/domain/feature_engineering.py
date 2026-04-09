"""Feature engineering for predicao of lambda via ML.

Funcoes puras that constroem vetores of features from dicts/DataFrames
of estatisticas of times. Zero deps of infra.

**v1.5.0 — 24 features (expandido of 12):**

Layout:
  0-5:   team_goals_for, team_goals_against, team_xg_for, team_xg_against,
         team_corners_for, team_cards_for
  6-9:   opp_goals_for, opp_goals_against, opp_xg_for, opp_xg_against
  10-11: league_goals_per_team, is_home
  12:    elo_diff (team_elo - opp_elo)
  13:    team_goal_diff_ema
  14:    team_xg_overperf
  15:    team_xga_overperf
  16:    team_creation_index
  17:    team_defensive_intensity
  18:    team_touches_per_match
  19:    team_rest_days
  20:    opp_creation_index
  21:    opp_defensive_intensity
  22:    opp_goal_diff_ema
  23:    opp_rest_days
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_DIM = 59

FEATURE_NAMES = [
    # 0-11: existing v1.3.0 (12)
    "team_goals_for", "team_goals_against",
    "team_xg_for", "team_xg_against",
    "team_corners_for", "team_cards_for",
    "opp_goals_for", "opp_goals_against",
    "opp_xg_for", "opp_xg_against",
    "league_goals_per_team", "is_home",
    # 12-23: v1.5.0 rich features (12)
    "elo_diff",
    "team_goal_diff_ema",
    "team_xg_overperf",
    "team_xga_overperf",
    "team_creation_index",
    "team_defensive_intensity",
    "team_touches_per_match",
    "team_rest_days",
    "opp_creation_index",
    "opp_defensive_intensity",
    "opp_goal_diff_ema",
    "opp_rest_days",
    # 24-39: v1.6.0 context features (16)
    "team_coach_win_rate",
    "team_games_since_coach_change",
    "team_coach_change_recent",
    "team_key_players_out",
    "team_xg_contribution_missing",
    "team_games_last_7d",
    "team_games_next_7d",
    "team_position",
    "opp_coach_win_rate",
    "opp_games_since_coach_change",
    "opp_coach_change_recent",
    "opp_key_players_out",
    "opp_xg_contribution_missing",
    "opp_position",
    "position_gap",
    "both_in_relegation",
    # 40-47: v1.8.0 playing style features (8)
    "team_finishing_efficiency",
    "team_sot_rate",
    "team_gk_quality",
    "team_possession_avg",
    "opp_finishing_efficiency",
    "opp_sot_rate",
    "opp_gk_quality",
    "opp_possession_avg",
    # 48-55: v1.10.0 H2H + Referee (8)
    "h2h_home_win_rate",
    "h2h_away_win_rate",
    "h2h_draw_rate",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",
    "ref_cards_per_game",
    "ref_strictness",
    "ref_experience",
    # 56-58: v1.13.0 Market-implied probs (3)
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
]


def build_team_features(
    team_stats: dict,
    opponent_stats: dict,
    league_avg: dict,
    is_home: bool,
) -> np.ndarray:
    """Returns array of FEATURE_DIM (40) with defaults seguros for v1.6.0.

    Backward compat: if chamado without Elo/rest/context, preenche defaults.
    """
    return build_context_aware_features(
        team_stats, opponent_stats, league_avg, is_home,
        team_elo=1500.0, opp_elo=1500.0,
        team_rest_days=7, opp_rest_days=7,
        team_context=None, opp_context=None,
    )


def build_rich_team_features(
    team_stats: dict,
    opponent_stats: dict,
    league_avg: dict,
    is_home: bool,
    team_elo: float = 1500.0,
    opp_elo: float = 1500.0,
    team_rest_days: int = 7,
    opp_rest_days: int = 7,
) -> np.ndarray:
    """Builds vetor of 24 features for predicao ML.

    Parameters
    ----------
    team_stats : dict
        Agregados of the time atacante. Chaves esperadas:
        goals_for, goals_against, xg_for, xg_against, corners_for,
        cards_for, goal_diff_ema, xg_overperf, xga_overperf,
        creation_index, defensive_intensity, touches_per_match.
    opponent_stats : dict
        Mesmas chaves of the adversario.
    league_avg : dict
        goals_per_team, corners_per_team.
    is_home : bool
    team_elo, opp_elo : float
        Ratings Elo PRE-match.
    team_rest_days, opp_rest_days : int
        Dias since last jogo of each time.

    Returns
    -------
    np.ndarray
        Array (24,) of float64.
    """
    league_goals = league_avg.get("goals_per_team", 1.3)

    features = np.array([
        # 0-5: team stats basicas
        float(team_stats.get("goals_for", league_goals)),
        float(team_stats.get("goals_against", league_goals)),
        float(team_stats.get("xg_for", league_goals)),
        float(team_stats.get("xg_against", league_goals)),
        float(team_stats.get("corners_for", 5.0)),
        float(team_stats.get("cards_for", 2.0)),
        # 6-9: opp stats basicas
        float(opponent_stats.get("goals_for", league_goals)),
        float(opponent_stats.get("goals_against", league_goals)),
        float(opponent_stats.get("xg_for", league_goals)),
        float(opponent_stats.get("xg_against", league_goals)),
        # 10-11: contexto
        float(league_goals),
        1.0 if is_home else 0.0,
        # 12: rating system
        float(team_elo - opp_elo),
        # 13-15: team form
        float(team_stats.get("goal_diff_ema", 0.0)),
        float(team_stats.get("xg_overperf", 0.0)),
        float(team_stats.get("xga_overperf", 0.0)),
        # 16-19: team advanced
        float(team_stats.get("creation_index", 0.0)),
        float(team_stats.get("defensive_intensity", 0.0)),
        float(team_stats.get("touches_per_match", 500.0)),
        float(team_rest_days),
        # 20-23: opp advanced
        float(opponent_stats.get("creation_index", 0.0)),
        float(opponent_stats.get("defensive_intensity", 0.0)),
        float(opponent_stats.get("goal_diff_ema", 0.0)),
        float(opp_rest_days),
    ], dtype=np.float64)

    return features


def build_context_aware_features(
    team_stats: dict,
    opponent_stats: dict,
    league_avg: dict,
    is_home: bool,
    team_elo: float = 1500.0,
    opp_elo: float = 1500.0,
    team_rest_days: int = 7,
    opp_rest_days: int = 7,
    team_context: dict | None = None,
    opp_context: dict | None = None,
    team_style: dict | None = None,
    opp_style: dict | None = None,
    h2h_features: dict | None = None,
    referee_features: dict | None = None,
    market_probs: dict | None = None,
) -> np.ndarray:
    """Builds vetor of FEATURE_DIM features.

    Parameters
    ----------
    team_stats, opponent_stats, league_avg, is_home, team_elo, opp_elo,
    team_rest_days, opp_rest_days :
        Mesma semantica of build_rich_team_features.
    team_context : dict | None
        {coach, injuries, fixtures, position} dicts of the team.
    opp_context : dict | None
        Mesmo pro adversario.

    Returns
    -------
    np.ndarray
        Array (40,) of float64.
    """
    from football_moneyball.domain.context_features import (
        coach_features, fixture_features, injury_features, position_features,
    )

    # Primeiras 24 features (v1.5.0)
    base = build_rich_team_features(
        team_stats, opponent_stats, league_avg, is_home,
        team_elo, opp_elo, team_rest_days, opp_rest_days,
    )

    # Team context
    tc = team_context or {}
    t_coach = coach_features(tc.get("coach"))
    t_inj = injury_features(tc.get("injuries"))
    t_fix = fixture_features(
        tc.get("fixtures", {}).get("games_last_7d", 0),
        tc.get("fixtures", {}).get("games_next_7d", 0),
    )
    t_pos = position_features(tc.get("position"))

    # Opp context
    oc = opp_context or {}
    o_coach = coach_features(oc.get("coach"))
    o_inj = injury_features(oc.get("injuries"))

    # team_position vs opp_position vs gap (from one of them)
    home_pos = t_pos.get("home_position", 10)
    away_pos = t_pos.get("away_position", 10)
    # if is_home, team_position = home_pos, else away_pos
    team_pos = home_pos if is_home else away_pos
    opp_pos = away_pos if is_home else home_pos

    context_features = np.array([
        # 24-31: team context (8)
        t_coach["coach_win_rate"],
        float(t_coach["games_since_change"]),
        float(t_coach["coach_change_recent"]),
        float(t_inj["key_players_out"]),
        t_inj["xg_contribution_missing"],
        float(t_fix["games_last_7d"]),
        float(t_fix["games_next_7d"]),
        float(team_pos),
        # 32-37: opp context (6)
        o_coach["coach_win_rate"],
        float(o_coach["games_since_change"]),
        float(o_coach["coach_change_recent"]),
        float(o_inj["key_players_out"]),
        o_inj["xg_contribution_missing"],
        float(opp_pos),
        # 38-39: match context (2)
        float(t_pos["position_gap"]),
        float(t_pos["both_in_relegation"]),
    ], dtype=np.float64)

    # v1.8.0 — Playing style features (8)
    ts = team_style or {}
    os_ = opp_style or {}
    style_features = np.array([
        # 40-43: team style (4)
        float(ts.get("finishing_efficiency", 0.35)),
        float(ts.get("sot_rate", 0.35)),
        float(ts.get("gk_quality", 0.0)),
        float(ts.get("possession_avg", 50.0)),
        # 44-47: opp style (4)
        float(os_.get("finishing_efficiency", 0.35)),
        float(os_.get("sot_rate", 0.35)),
        float(os_.get("gk_quality", 0.0)),
        float(os_.get("possession_avg", 50.0)),
    ], dtype=np.float64)

    # v1.10.0 — H2H + Referee features (8)
    h2h = h2h_features or {}
    ref = referee_features or {}
    extra_features = np.array([
        # 48-52: H2H (5)
        float(h2h.get("h2h_home_win_rate", 0.33)),
        float(h2h.get("h2h_away_win_rate", 0.33)),
        float(h2h.get("h2h_draw_rate", 0.25)),
        float(h2h.get("h2h_home_goals_avg", 1.3)),
        float(h2h.get("h2h_away_goals_avg", 1.3)),
        # 53-55: Referee (3)
        float(ref.get("ref_cards_per_game", 4.2)),
        float(ref.get("ref_strictness", 0.0)),
        float(ref.get("ref_experience", 0.0)),
    ], dtype=np.float64)

    # v1.13.0 — Market-implied probs (3)
    mkt = market_probs or {}
    market_features = np.array([
        float(mkt.get("market_home_prob", 0.40)),
        float(mkt.get("market_draw_prob", 0.28)),
        float(mkt.get("market_away_prob", 0.32)),
    ], dtype=np.float64)

    return np.concatenate([base, context_features, style_features, extra_features, market_features])


def _team_rolling_stats(
    past_matches: pd.DataFrame,
    team: str,
    last_n: int = 5,
    decay: float = 0.85,
) -> dict:
    """Compute agregados of the last N jogos of the time.

    Returns dict com:
        goals_for, goals_against, xg_for, xg_against, corners_for, cards_for,
        goal_diff_ema, xg_overperf, xga_overperf,
        creation_index, defensive_intensity, touches_per_match.
    """
    defaults = {
        "goals_for": 1.3, "goals_against": 1.3,
        "xg_for": 1.3, "xg_against": 1.3,
        "corners_for": 5.0, "cards_for": 2.0,
        "goal_diff_ema": 0.0,
        "xg_overperf": 0.0, "xga_overperf": 0.0,
        "creation_index": 0.0, "defensive_intensity": 0.0,
        "touches_per_match": 500.0,
    }

    if past_matches.empty or "home_team" not in past_matches.columns:
        return defaults

    mask = (past_matches["home_team"] == team) | (past_matches["away_team"] == team)
    team_matches = past_matches[mask].tail(last_n)

    if team_matches.empty:
        return defaults

    # Vetores by perspectiva of the time
    gf, ga, xf, xa_against = [], [], [], []
    corners_for, cards_for = [], []
    xa_team, key_passes_team = [], []
    tackles_team, interceptions_team, recoveries_team, touches_team = [], [], [], []

    for _, row in team_matches.iterrows():
        is_home = row["home_team"] == team
        if is_home:
            gf.append(float(row.get("home_goals") or 0))
            ga.append(float(row.get("away_goals") or 0))
            xf.append(float(row.get("home_xg") or 0))
            xa_against.append(float(row.get("away_xg") or 0))
            corners_for.append(float(row.get("home_corners") or 0))
            cards_for.append(float(row.get("home_cards") or 0))
            xa_team.append(float(row.get("home_xa") or 0))
            key_passes_team.append(float(row.get("home_key_passes") or 0))
            tackles_team.append(float(row.get("home_tackles") or 0))
            interceptions_team.append(float(row.get("home_interceptions") or 0))
            recoveries_team.append(float(row.get("home_recoveries") or 0))
            touches_team.append(float(row.get("home_touches") or 0))
        else:
            gf.append(float(row.get("away_goals") or 0))
            ga.append(float(row.get("home_goals") or 0))
            xf.append(float(row.get("away_xg") or 0))
            xa_against.append(float(row.get("home_xg") or 0))
            corners_for.append(float(row.get("away_corners") or 0))
            cards_for.append(float(row.get("away_cards") or 0))
            xa_team.append(float(row.get("away_xa") or 0))
            key_passes_team.append(float(row.get("away_key_passes") or 0))
            tackles_team.append(float(row.get("away_tackles") or 0))
            interceptions_team.append(float(row.get("away_interceptions") or 0))
            recoveries_team.append(float(row.get("away_recoveries") or 0))
            touches_team.append(float(row.get("away_touches") or 0))

    n = len(gf)
    # Medias simples
    mean_gf = float(np.mean(gf))
    mean_ga = float(np.mean(ga))
    mean_xf = float(np.mean(xf))
    mean_xa = float(np.mean(xa_against))

    # EMA goal diff (mais recente pesa mais)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum() if weights.sum() > 0 else 1
    goal_diffs = np.array(gf) - np.array(ga)
    goal_diff_ema = float(np.dot(goal_diffs, weights))

    # xG overperformance (goals - xG esperados) by jogo
    xg_overperf = (sum(gf) - sum(xf)) / n if n > 0 else 0.0
    xga_overperf = (sum(ga) - sum(xa_against)) / n if n > 0 else 0.0

    # Creation index: xa/90 + key_passes*0.05/90 (approx minutos = 90)
    sum_xa = sum(xa_team)
    sum_kp = sum(key_passes_team)
    creation_index = (sum_xa + sum_kp * 0.05) / n if n > 0 else 0.0

    # Defensive intensity: tackles + interceptions + recoveries by jogo
    defensive_intensity = (
        (sum(tackles_team) + sum(interceptions_team) + sum(recoveries_team)) / n
        if n > 0 else 0.0
    )

    # Touches by jogo
    touches_per_match = sum(touches_team) / n if n > 0 else 500.0

    return {
        "goals_for": mean_gf, "goals_against": mean_ga,
        "xg_for": mean_xf, "xg_against": mean_xa,
        "corners_for": float(np.mean(corners_for)),
        "cards_for": float(np.mean(cards_for)),
        "goal_diff_ema": goal_diff_ema,
        "xg_overperf": xg_overperf,
        "xga_overperf": xga_overperf,
        "creation_index": creation_index,
        "defensive_intensity": defensive_intensity,
        "touches_per_match": touches_per_match,
    }


def _compute_rest_days(
    past_matches: pd.DataFrame,
    team: str,
    match_date: str,
    default: int = 7,
) -> int:
    """Dias since last jogo of the time antes of match_date.

    Fallback: `default` (7 days) if without history.
    """
    if past_matches.empty or not match_date:
        return default

    mask = (past_matches["home_team"] == team) | (past_matches["away_team"] == team)
    team_past = past_matches[mask]
    if team_past.empty or "match_date" not in team_past.columns:
        return default

    try:
        target = pd.to_datetime(match_date).date()
        last_date_val = pd.to_datetime(team_past["match_date"].iloc[-1]).date()
        diff = (target - last_date_val).days
        return int(max(1, diff))
    except Exception:
        return default


def _compute_h2h_from_past(
    past: pd.DataFrame, home: str, away: str, last_n: int = 5,
) -> dict[str, float]:
    """Compute H2H features of the DataFrame passado (leak-proof)."""
    from football_moneyball.domain.h2h_features import compute_h2h_features
    h2h_matches = past[
        ((past["home_team"] == home) & (past["away_team"] == away))
        | ((past["home_team"] == away) & (past["away_team"] == home))
    ].tail(last_n)
    history = [
        {
            "home_team": r["home_team"], "away_team": r["away_team"],
            "home_goals": r.get("home_goals", 0) or 0,
            "away_goals": r.get("away_goals", 0) or 0,
        }
        for _, r in h2h_matches.iterrows()
    ]
    return compute_h2h_features(history, home, away)


def build_training_dataset(
    matches_with_stats: pd.DataFrame,
    target: str = "goals",
    last_n: int = 5,
    min_prior: int = 3,
    match_referees: dict[int, dict] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds (X, y) for training usando APENAS data earlier a each match.

    v1.10.0 — inclui H2H (leak-proof) + referee features.

    Parameters
    ----------
    match_referees : dict[int, dict] | None
        Mapping match_id -> referee_stats dict. If None, ref features usam defaults.
    """
    from football_moneyball.domain.elo import compute_elo_timeline
    from football_moneyball.domain.referee_features import compute_referee_features

    target_col_home = f"home_{target}"
    target_col_away = f"away_{target}"

    df = matches_with_stats.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # League averages globais
    league_avg = {
        "goals_per_team": float(
            (df["home_goals"].fillna(0).mean() + df["away_goals"].fillna(0).mean()) / 2
        ) if not df.empty else 1.3,
        "corners_per_team": float(
            (df["home_corners"].fillna(5).mean() + df["away_corners"].fillna(5).mean()) / 2
        ) if "home_corners" in df.columns and not df.empty else 5.0,
    }

    # Precompute Elo timeline (evita leak — Elo pre-match, nao inclui jogo atual)
    elo_timeline = compute_elo_timeline(df)

    X_rows = []
    y_rows = []

    for idx, row in df.iterrows():
        past = df.iloc[:idx]
        home = row["home_team"]
        away = row["away_team"]
        match_id = int(row["match_id"])
        match_date = str(row.get("match_date", ""))

        n_home_prior = ((past["home_team"] == home) | (past["away_team"] == home)).sum()
        n_away_prior = ((past["home_team"] == away) | (past["away_team"] == away)).sum()
        if n_home_prior < min_prior or n_away_prior < min_prior:
            continue

        home_hist = _team_rolling_stats(past, home, last_n)
        away_hist = _team_rolling_stats(past, away, last_n)

        # Elo PRE-match (compute_elo_timeline armazena value ANTES of the jogo)
        home_elo = elo_timeline.get((match_id, home), 1500.0)
        away_elo = elo_timeline.get((match_id, away), 1500.0)

        # Rest days
        home_rest = _compute_rest_days(past, home, match_date)
        away_rest = _compute_rest_days(past, away, match_date)

        # v1.10.0: H2H + Referee features (leak-proof)
        h2h_home = _compute_h2h_from_past(past, home, away, last_n=5)
        h2h_away = _compute_h2h_from_past(past, away, home, last_n=5)
        ref_stats = (match_referees or {}).get(match_id)
        ref_feats = compute_referee_features(ref_stats)

        # Sample 1: home ataca
        X_rows.append(build_context_aware_features(
            home_hist, away_hist, league_avg, is_home=True,
            team_elo=home_elo, opp_elo=away_elo,
            team_rest_days=home_rest, opp_rest_days=away_rest,
            team_context=None, opp_context=None,
            h2h_features=h2h_home, referee_features=ref_feats,
        ))
        y_rows.append(float(row.get(target_col_home, 0) or 0))

        # Sample 2: away ataca
        X_rows.append(build_context_aware_features(
            away_hist, home_hist, league_avg, is_home=False,
            team_elo=away_elo, opp_elo=home_elo,
            team_rest_days=away_rest, opp_rest_days=home_rest,
            team_context=None, opp_context=None,
            h2h_features=h2h_away, referee_features=ref_feats,
        ))
        y_rows.append(float(row.get(target_col_away, 0) or 0))

    if not X_rows:
        return np.zeros((0, FEATURE_DIM)), np.zeros(0)

    return np.vstack(X_rows), np.array(y_rows)
