"""Modulo of previsao of matches via Dixon-Coles simplificado.

Compute parameters dinamicos from TODOS os jogos of the season
(with decaimento exponencial — jogos recentes valem mais), estimates
attack/defense strength by time, aplica regression a mean, and simula
resultados via Monte Carlo + Poisson.

Zero constantes hardcoded — tudo calculado of the DataFrame passado.

Referencia: Dixon & Coles (1997) — Modelling Association Football
Scores and Inefficiencies in the Football Betting Market.
"""

from __future__ import annotations

from math import factorial

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. League Averages (dynamic, time-weighted)
# ---------------------------------------------------------------------------

def calculate_league_averages(
    all_match_data: pd.DataFrame,
    decay: float = 0.95,
) -> dict:
    """Compute means of the liga with decaimento exponencial.

    Jogos mais recentes pesam mais. Parametros sao recalculados
    a each previsao — nada eh hardcoded.

    Parameters
    ----------
    all_match_data : pd.DataFrame
        Todas as matches: match_id, team, goals, xg, is_home.
        Uma row by time by match.
    decay : float
        Fator of decaimento by matchday (0.95 = jogo 10 rodadas atras
        pesa 0.95^10 = 60% of a jogo recente).

    Returns
    -------
    dict
        avg_xg, avg_goals, avg_xg_home, avg_xg_away, home_advantage,
        n_matches.
    """
    if all_match_data.empty:
        return {
            "avg_xg": 1.25, "avg_goals": 1.30,
            "avg_xg_home": 1.40, "avg_xg_away": 1.10,
            "home_advantage": 0.20, "n_matches": 0,
        }

    # Ordenar by match_id (proxy cronologico) and atribuir weights
    match_ids = sorted(all_match_data["match_id"].unique())
    n = len(match_ids)
    match_weights = {mid: decay ** (n - 1 - i) for i, mid in enumerate(match_ids)}

    df = all_match_data.copy()
    df["weight"] = df["match_id"].map(match_weights)

    # Medias weighted
    total_weight = df.groupby("match_id")["weight"].first().sum()
    if total_weight <= 0:
        total_weight = 1.0

    avg_xg = (df["xg"] * df["weight"]).sum() / (total_weight * 2)  # 2 times by jogo
    avg_goals = (df["goals"] * df["weight"]).sum() / (total_weight * 2)

    # Home vs Away
    home = df[df["is_home"] == True]  # noqa: E712
    away = df[df["is_home"] == False]  # noqa: E712

    home_weight = home["weight"].sum()
    away_weight = away["weight"].sum()

    avg_xg_home = (home["xg"] * home["weight"]).sum() / max(home_weight, 1)
    avg_xg_away = (away["xg"] * away["weight"]).sum() / max(away_weight, 1)

    home_advantage = avg_xg_home - avg_xg_away

    return {
        "avg_xg": float(max(avg_xg, 0.5)),
        "avg_goals": float(max(avg_goals, 0.5)),
        "avg_xg_home": float(avg_xg_home),
        "avg_xg_away": float(avg_xg_away),
        "home_advantage": float(home_advantage),
        "n_matches": n,
    }


# ---------------------------------------------------------------------------
# 2. Team Strength (ataque and defesa relativos a liga)
# ---------------------------------------------------------------------------

def calculate_team_strength(
    team_matches: pd.DataFrame,
    all_match_data: pd.DataFrame,
    league_avgs: dict,
    decay: float = 0.85,
) -> dict:
    """Compute forca offensive and defensive of a time relativa a liga.

    attack_strength > 1.0 = ataque acima of the mean
    defense_strength < 1.0 = defesa boa (sofre menos that mean)

    Parameters
    ----------
    team_matches : pd.DataFrame
        Matches of the time: match_id, team, goals, xg, is_home.
    all_match_data : pd.DataFrame
        Todas as matches of the liga (for calcular xGA).
    league_avgs : dict
        Output of calculate_league_averages.
    decay : float
        Decaimento by jogo (mais agressivo that liga).

    Returns
    -------
    dict
        attack_strength, defense_strength, xg_avg, xga_avg,
        goals_total, xg_total, matches.
    """
    if team_matches.empty:
        return {
            "attack_strength": 1.0, "defense_strength": 1.0,
            "xg_avg": league_avgs["avg_xg"], "xga_avg": league_avgs["avg_xg"],
            "goals_total": 0, "xg_total": 0.0, "matches": 0,
        }

    team_name = team_matches["team"].iloc[0]

    # Pesos by recencia
    match_ids = sorted(team_matches["match_id"].unique())
    n = len(match_ids)
    weights = {mid: decay ** (n - 1 - i) for i, mid in enumerate(match_ids)}

    tm = team_matches.copy()
    tm["weight"] = tm["match_id"].map(weights)
    total_w = tm["weight"].sum()

    # xG medio weighted of the time
    xg_avg = (tm["xg"] * tm["weight"]).sum() / max(total_w, 1)

    # xGA medio (xG sofrido = xG of the adversario naquela match)
    xga_records = []
    for mid in match_ids:
        opp_data = all_match_data[
            (all_match_data["match_id"] == mid) &
            (all_match_data["team"] != team_name)
        ]
        if not opp_data.empty:
            xga_records.append({
                "match_id": mid,
                "xga": opp_data["xg"].sum(),
                "weight": weights[mid],
            })

    if xga_records:
        xga_df = pd.DataFrame(xga_records)
        xga_avg = (xga_df["xga"] * xga_df["weight"]).sum() / xga_df["weight"].sum()
    else:
        xga_avg = league_avgs["avg_xg"]

    # Strength relativa
    avg_xg_league = league_avgs["avg_xg"]
    attack_strength = xg_avg / avg_xg_league if avg_xg_league > 0 else 1.0
    defense_strength = xga_avg / avg_xg_league if avg_xg_league > 0 else 1.0

    return {
        "attack_strength": float(attack_strength),
        "defense_strength": float(defense_strength),
        "xg_avg": float(xg_avg),
        "xga_avg": float(xga_avg),
        "goals_total": int(tm["goals"].sum()),
        "xg_total": float(tm["xg"].sum()),
        "matches": n,
    }


# ---------------------------------------------------------------------------
# 3. Shot Quality (proxy for xT)
# ---------------------------------------------------------------------------

def calculate_xg_quality(
    shot_xgs: list[float],
    league_avg_shot_xg: float = 0.10,
) -> float:
    """Fator of qualidade of shots relativo a mean.

    Times with muitos shots of alto xG (big chances) criam mais
    perigo real that times with muitos shots of baixo xG.

    Parameters
    ----------
    shot_xgs : list[float]
        xG of each shot individual of the time (last N jogos).
    league_avg_shot_xg : float
        If 0, usa mean of the proprios shots as baseline.

    Returns
    -------
    float
        Fator multiplicativo (1.0 = medio, >1.0 = shots melhores).
    """
    if not shot_xgs or len(shot_xgs) < 3:
        return 1.0

    avg_shot = float(np.mean(shot_xgs))

    if league_avg_shot_xg <= 0:
        return 1.0

    # Suavizar: nao deixar o fator ser extremo
    raw_factor = avg_shot / league_avg_shot_xg
    # Clamp between 0.7 and 1.3
    return float(np.clip(raw_factor, 0.7, 1.3))


# ---------------------------------------------------------------------------
# 4. Regression to Mean
# ---------------------------------------------------------------------------

def apply_regression_to_mean(
    xg_estimate: float,
    team_goals: int,
    team_xg_total: float,
    matches_played: int,
    k: float = 15.0,
) -> float:
    """Puxa overperformers in direcao a mean.

    Regression factor diminui with mais jogos — amostra grande eh
    mais confiavel, entao regride menos.

    factor = k / (k + matches_played)
    With k=15: 5 jogos → 75% regression, 15 jogos → 50%, 30 jogos → 33%

    Parameters
    ----------
    xg_estimate : float
        xG estimado by the model.
    team_goals : int
        Gols reais scored in the season.
    team_xg_total : float
        xG total in the season.
    matches_played : int
        Numero of matches jogadas.
    k : float
        Constante of regression (maior = mais regression).

    Returns
    -------
    float
        xG ajustado by the regression.
    """
    if matches_played <= 0 or team_xg_total <= 0:
        return max(xg_estimate, 0.1)

    overperformance_per_game = (team_goals - team_xg_total) / matches_played
    regression_factor = k / (k + matches_played)
    adjustment = -overperformance_per_game * regression_factor

    return float(max(xg_estimate + adjustment, 0.1))


# ---------------------------------------------------------------------------
# 5. Monte Carlo Simulation (mantido of the v0.4.0)
# ---------------------------------------------------------------------------

def simulate_match(
    home_xg: float,
    away_xg: float,
    n_simulations: int = 10_000,
    seed: int | None = None,
    dixon_coles_rho: float | None = -0.10,
    score_method: str = "dixon-coles",
    bivariate_lambda3: float = 0.10,
) -> dict:
    """Simula a match N vezes via Monte Carlo.

    Suporta 3 motores of sampling:
    - ``"bivariate"``: bivariate Poisson diagonal-inflated (Karlis & Ntzoufras 2003)
    - ``"dixon-coles"``: Poisson independente + correcao tau (Dixon & Coles 1997)
    - ``"poisson"``: Poisson independente puro

    Parameters
    ----------
    home_xg, away_xg : float
        xG esperado of each time.
    n_simulations : int
        Numero of simulacoes.
    seed : int, optional
        Seed for reprodutibilidade.
    dixon_coles_rho : float | None
        Parametro ρ (Dixon-Coles). Ignorado if score_method != "dixon-coles".
    score_method : str
        Motor of sampling: "bivariate", "dixon-coles", "poisson".
    bivariate_lambda3 : float
        Parametro λ3 (bivariate Poisson). Ignorado if score_method != "bivariate".

    Returns
    -------
    dict
        Probabilidades of todos os mercados.
    """
    rng = np.random.default_rng(seed)

    if score_method == "bivariate":
        from football_moneyball.domain.calibration import sample_scores_bivariate
        home_goals, away_goals = sample_scores_bivariate(
            home_xg, away_xg, bivariate_lambda3, n_simulations, seed=seed,
        )
    elif score_method == "dixon-coles" and dixon_coles_rho is not None:
        from football_moneyball.domain.calibration import sample_scores_dixon_coles
        home_goals, away_goals = sample_scores_dixon_coles(
            home_xg, away_xg, dixon_coles_rho, n_simulations, seed=seed,
        )
    else:
        home_goals = rng.poisson(home_xg, n_simulations)
        away_goals = rng.poisson(away_xg, n_simulations)
    total_goals = home_goals + away_goals

    home_wins = (home_goals > away_goals).sum()
    draws = (home_goals == away_goals).sum()
    away_wins = (home_goals < away_goals).sum()

    over_05 = (total_goals > 0.5).sum()
    over_15 = (total_goals > 1.5).sum()
    over_25 = (total_goals > 2.5).sum()
    over_35 = (total_goals > 3.5).sum()

    btts = ((home_goals > 0) & (away_goals > 0)).sum()

    score_counts: dict[str, int] = {}
    for h, a in zip(home_goals, away_goals):
        key = f"{h}x{a}"
        score_counts[key] = score_counts.get(key, 0) + 1

    most_likely = max(score_counts, key=score_counts.get) if score_counts else "0x0"

    score_matrix = {
        k: round(v / n_simulations, 4)
        for k, v in sorted(score_counts.items(), key=lambda x: -x[1])[:10]
    }

    n = n_simulations
    return {
        "home_xg": round(home_xg, 3),
        "away_xg": round(away_xg, 3),
        "home_win_prob": round(home_wins / n, 4),
        "draw_prob": round(draws / n, 4),
        "away_win_prob": round(away_wins / n, 4),
        "over_05": round(over_05 / n, 4),
        "over_15": round(over_15 / n, 4),
        "over_25": round(over_25 / n, 4),
        "over_35": round(over_35 / n, 4),
        "btts_prob": round(btts / n, 4),
        "most_likely_score": most_likely,
        "score_matrix": score_matrix,
        "simulations": n_simulations,
    }


# ---------------------------------------------------------------------------
# 6. Pipeline Completo
# ---------------------------------------------------------------------------

def predict_match(
    home_team: str,
    away_team: str,
    all_match_data: pd.DataFrame,
    home_shots: list[float] | None = None,
    away_shots: list[float] | None = None,
    n_simulations: int = 10_000,
    seed: int | None = None,
    dixon_coles_rho: float | None = -0.10,
    score_method: str = "dixon-coles",
    bivariate_lambda3: float = 0.10,
) -> dict:
    """Pipeline complete of previsao — calcula TUDO of the DataFrame.

    Nenhuma constante hardcoded. Todos os parameters sao derivados
    dinamicamente of the data passados.

    Parameters
    ----------
    home_team, away_team : str
        Nomes of the times.
    all_match_data : pd.DataFrame
        TODOS os jogos of the season with colunas:
        match_id, team, goals, xg, is_home.
    home_shots, away_shots : list[float], optional
        xG of each shot individual (last jogos).
    n_simulations : int
        Numero of simulacoes Monte Carlo.
    seed : int, optional
    dixon_coles_rho : float | None
        Parametro of correcao Dixon-Coles. None = Poisson independente.
        Seed for reprodutibilidade.

    Returns
    -------
    dict
        Probabilidades + metadados of the pipeline.
    """
    # 0. Fuzzy match team names (odds API returns without acentos)
    home_team = _fuzzy_match_team(home_team, all_match_data["team"].unique())
    away_team = _fuzzy_match_team(away_team, all_match_data["team"].unique())

    # 1. Medias of the liga (dinamicas, weighted)
    league = calculate_league_averages(all_match_data)

    # 2. Forca of each time
    home_data = all_match_data[all_match_data["team"] == home_team]
    away_data = all_match_data[all_match_data["team"] == away_team]

    home_str = calculate_team_strength(home_data, all_match_data, league)
    away_str = calculate_team_strength(away_data, all_match_data, league)

    # 3. xG esperado = league_avg × attack × opp_defense
    home_xg = league["avg_xg"] * home_str["attack_strength"] * away_str["defense_strength"]
    away_xg = league["avg_xg"] * away_str["attack_strength"] * home_str["defense_strength"]

    # 4. Qualidade of shots
    if home_shots:
        # Calcular league avg shot xG dinamicamente
        all_shots = (home_shots or []) + (away_shots or [])
        league_avg_shot = float(np.mean(all_shots)) if all_shots else 0.10
        home_xg *= calculate_xg_quality(home_shots, league_avg_shot)
    if away_shots:
        all_shots = (home_shots or []) + (away_shots or [])
        league_avg_shot = float(np.mean(all_shots)) if all_shots else 0.10
        away_xg *= calculate_xg_quality(away_shots, league_avg_shot)

    # 5. Regressao a mean
    home_xg = apply_regression_to_mean(
        home_xg, home_str["goals_total"], home_str["xg_total"], home_str["matches"]
    )
    away_xg = apply_regression_to_mean(
        away_xg, away_str["goals_total"], away_str["xg_total"], away_str["matches"]
    )

    # 6. Home advantage (dinamico)
    home_xg += league["home_advantage"]

    # Clamp
    home_xg = max(home_xg, 0.15)
    away_xg = max(away_xg, 0.15)

    # 7. Monte Carlo
    result = simulate_match(home_xg, away_xg, n_simulations, seed,
                            dixon_coles_rho=dixon_coles_rho,
                            score_method=score_method,
                            bivariate_lambda3=bivariate_lambda3)

    # Metadados of the pipeline
    result["pipeline"] = {
        "league_avg_xg": league["avg_xg"],
        "home_advantage": league["home_advantage"],
        "home_attack": home_str["attack_strength"],
        "home_defense": home_str["defense_strength"],
        "away_attack": away_str["attack_strength"],
        "away_defense": away_str["defense_strength"],
        "n_matches_league": league["n_matches"],
    }

    return result


# ---------------------------------------------------------------------------
# 6b. Pipeline Player-Aware (v1.1.0)
# ---------------------------------------------------------------------------

def predict_match_player_aware(
    home_team: str,
    away_team: str,
    all_match_data: pd.DataFrame,
    home_player_aggregates: pd.DataFrame,
    away_player_aggregates: pd.DataFrame,
    n_simulations: int = 10_000,
    seed: int | None = None,
    last_n: int = 5,
    dixon_coles_rho: float | None = -0.10,
    score_method: str = "dixon-coles",
    bivariate_lambda3: float = 0.10,
) -> dict:
    """Pipeline player-aware — λ derivado of the 11 titulares provaveis.

    Substitui o path team-level: in vez de `league_avg × attack_strength`,
    usa `Σ(xG/90 of the 11 titulares × weight)` as base of the lambda. Mantem
    todos os outros ajustes (defesa oposta, home advantage, regression).

    Parameters
    ----------
    home_team, away_team : str
        Nomes of the times.
    all_match_data : pd.DataFrame
        Data team-level for calcular home_advantage and defesa oposta.
        Colunas: match_id, team, goals, xg, is_home.
    home_player_aggregates, away_player_aggregates : pd.DataFrame
        Agregacoes by player of the last N jogos. Colunas:
        player_id, player_name, matches_played, minutes_total, xg_total.
    n_simulations : int
        Numero of simulacoes Monte Carlo.
    seed : int, optional
        Seed for reprodutibilidade.
    last_n : int
        Janela of referencia used in the probable_xi (default 5).

    Returns
    -------
    dict
        Mesmas chaves of the predict_match() + metadados player-aware:
        lineup_type, home_xi, away_xi, home_team_attack, away_team_attack.
    """
    from football_moneyball.domain.lineup_prediction import probable_xi
    from football_moneyball.domain.player_lambda import (
        team_lambda_from_players, summarize_xi,
    )

    # 0. Fuzzy match team names (odds API returns without acentos)
    home_team = _fuzzy_match_team(home_team, all_match_data["team"].unique())
    away_team = _fuzzy_match_team(away_team, all_match_data["team"].unique())

    # 1. Medias of the liga (for home_advantage)
    league = calculate_league_averages(all_match_data)

    # 2. Defesa of each time (team-level — player-level nao tem xGA)
    home_data = all_match_data[all_match_data["team"] == home_team]
    away_data = all_match_data[all_match_data["team"] == away_team]
    home_str = calculate_team_strength(home_data, all_match_data, league)
    away_str = calculate_team_strength(away_data, all_match_data, league)

    # 3. Probable XI of each time
    home_xi = probable_xi(home_player_aggregates, last_n_matches=last_n)
    away_xi = probable_xi(away_player_aggregates, last_n_matches=last_n)

    # 4. λ = Σ(xG/90 × weight) × opp_defense
    home_xg = team_lambda_from_players(home_xi, away_str["defense_strength"])
    away_xg = team_lambda_from_players(away_xi, home_str["defense_strength"])

    # 5. Regressao a mean (still a nivel of time)
    home_xg = apply_regression_to_mean(
        home_xg,
        home_str["goals_total"],
        home_str["xg_total"],
        home_str["matches"],
    )
    away_xg = apply_regression_to_mean(
        away_xg,
        away_str["goals_total"],
        away_str["xg_total"],
        away_str["matches"],
    )

    # 6. Home advantage (dinamico)
    home_xg += league["home_advantage"]

    # Clamp
    home_xg = max(home_xg, 0.15)
    away_xg = max(away_xg, 0.15)

    # 7. Monte Carlo
    result = simulate_match(home_xg, away_xg, n_simulations, seed,
                            dixon_coles_rho=dixon_coles_rho,
                            score_method=score_method,
                            bivariate_lambda3=bivariate_lambda3)

    # Metadados player-aware
    home_team_attack = float((home_xi["xg_per_90"] * home_xi["weight"]).sum()) if not home_xi.empty else 0.0
    away_team_attack = float((away_xi["xg_per_90"] * away_xi["weight"]).sum()) if not away_xi.empty else 0.0

    result["lineup_type"] = "probable-xi"
    result["model_version"] = "v1.10.0"
    result["home_xi"] = summarize_xi(home_xi)
    result["away_xi"] = summarize_xi(away_xi)
    result["pipeline"] = {
        "league_avg_xg": league["avg_xg"],
        "home_advantage": league["home_advantage"],
        "home_team_attack": round(home_team_attack, 3),
        "away_team_attack": round(away_team_attack, 3),
        "home_defense": home_str["defense_strength"],
        "away_defense": away_str["defense_strength"],
        "n_matches_league": league["n_matches"],
        "home_xi_size": len(home_xi),
        "away_xi_size": len(away_xi),
    }

    return result


# ---------------------------------------------------------------------------
# 7. Fuzzy team name matching
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Remove acentos and normaliza for comparacao."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _fuzzy_match_team(name: str, known_teams: list | np.ndarray) -> str:
    """Encontra o time mais next in the dataset by nome normalizado."""
    norm = _normalize_name(name)
    for team in known_teams:
        if _normalize_name(team) == norm:
            return team
    # Substring match
    for team in known_teams:
        if norm in _normalize_name(team) or _normalize_name(team) in norm:
            return team
    return name  # fallback: returns original


# ---------------------------------------------------------------------------
# 8. Helpers (mantidos)
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) for distribuicao Poisson with parameter lambda."""
    if lam <= 0 or k < 0:
        return 0.0
    return float(np.exp(-lam) * (lam ** k) / factorial(k))


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def estimate_team_xg(
    team_history: pd.DataFrame,
    opponent_history: pd.DataFrame,
    is_home: bool,
    n_games: int = 6,
    decay: float = 0.85,
    home_advantage: float = 0.30,
) -> float:
    """DEPRECATED: Usar predict_match() in vez disso.

    Mantido for backward compatibility with backtest v0.4.0.
    """
    recent_xg = team_history["xg"].head(n_games).values
    if len(recent_xg) == 0:
        return 1.0

    weights = np.array([decay ** i for i in range(len(recent_xg))])
    weights /= weights.sum()
    avg_xg = float(np.dot(recent_xg, weights))

    if not opponent_history.empty and "xg_against" in opponent_history.columns:
        opp_xga = opponent_history["xg_against"].head(n_games).mean()
        league_avg_xg = 1.25
        strength_factor = opp_xga / league_avg_xg if league_avg_xg > 0 else 1.0
        avg_xg *= strength_factor

    if is_home:
        avg_xg += home_advantage

    return max(avg_xg, 0.1)
