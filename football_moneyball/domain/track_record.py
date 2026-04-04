"""Logica pura de track record — resolve previsoes e calcula metricas de acuracia.

Modulo de dominio: nao importa SQLAlchemy, requests, ou qualquer infra.
Apenas opera sobre dicts e listas Python nativas.
"""

from __future__ import annotations

from collections import defaultdict


def resolve_prediction(
    pred: dict,
    actual_home_goals: int,
    actual_away_goals: int,
) -> dict:
    """Preenche campos de resolucao de uma previsao.

    Recebe a previsao (dict com probabilidades do modelo) e o placar real,
    retorna dict com campos preenchidos: actual_outcome, correct_1x2,
    correct_over_under, brier_score, status='resolved'.

    Parameters
    ----------
    pred : dict
        Previsao original com home_win_prob, draw_prob, away_win_prob,
        over_25_prob.
    actual_home_goals : int
        Gols do mandante.
    actual_away_goals : int
        Gols do visitante.

    Returns
    -------
    dict
        Campos de resolucao para atualizar no registro.
    """
    # Resultado real
    if actual_home_goals > actual_away_goals:
        actual_outcome = "home"
    elif actual_home_goals < actual_away_goals:
        actual_outcome = "away"
    else:
        actual_outcome = "draw"

    # Resultado previsto (maior probabilidade)
    home_prob = float(pred.get("home_win_prob", 0) or 0)
    draw_prob = float(pred.get("draw_prob", 0) or 0)
    away_prob = float(pred.get("away_win_prob", 0) or 0)

    probs = {"home": home_prob, "draw": draw_prob, "away": away_prob}
    predicted_outcome = max(probs, key=probs.get)

    correct_1x2 = predicted_outcome == actual_outcome

    # Over/Under 2.5
    total_goals = actual_home_goals + actual_away_goals
    actual_over = total_goals > 2.5
    over_prob = float(pred.get("over_25_prob", 0) or 0)
    predicted_over = over_prob > 0.5
    correct_over_under = predicted_over == actual_over

    # Brier score (para 1X2 — multiclass)
    actual_vec = [
        1.0 if actual_outcome == "home" else 0.0,
        1.0 if actual_outcome == "draw" else 0.0,
        1.0 if actual_outcome == "away" else 0.0,
    ]
    pred_vec = [home_prob, draw_prob, away_prob]
    brier = sum((p - a) ** 2 for p, a in zip(pred_vec, actual_vec)) / len(actual_vec)

    return {
        "actual_home_goals": actual_home_goals,
        "actual_away_goals": actual_away_goals,
        "actual_outcome": actual_outcome,
        "correct_1x2": correct_1x2,
        "correct_over_under": correct_over_under,
        "brier_score": round(brier, 6),
        "status": "resolved",
    }


def resolve_value_bet(
    bet: dict,
    actual_outcome: str,
    actual_total_goals: int,
) -> dict:
    """Resolve uma value bet com base no resultado real.

    Parameters
    ----------
    bet : dict
        Value bet com market, outcome, best_odds, kelly_stake.
    actual_outcome : str
        'home', 'draw', ou 'away'.
    actual_total_goals : int
        Total de gols da partida.

    Returns
    -------
    dict
        Campos de resolucao: won, profit.
    """
    market = bet.get("market", "")
    bet_outcome = bet.get("outcome", "")
    odds = float(bet.get("best_odds", 0) or 0)
    stake = float(bet.get("kelly_stake", 0) or 0)

    won = False

    if market == "h2h":
        # Mapear outcome para home/draw/away
        outcome_lower = bet_outcome.lower()
        if outcome_lower == "draw":
            won = actual_outcome == "draw"
        else:
            home_team = bet.get("home_team", "")
            if bet_outcome == home_team:
                won = actual_outcome == "home"
            else:
                won = actual_outcome == "away"
    elif market in ("totals", "over_under"):
        if "over" in bet_outcome.lower():
            won = actual_total_goals > 2.5
        elif "under" in bet_outcome.lower():
            won = actual_total_goals <= 2.5
    elif market == "btts":
        # Both teams to score — nao temos info suficiente aqui
        # mas podemos checar se ambos marcaram
        pass

    profit = (odds - 1) * stake if won else -stake

    return {
        "won": won,
        "profit": round(profit, 2),
    }


def calculate_track_record(predictions: list[dict]) -> dict:
    """Calcula resumo do track record a partir do historico de previsoes.

    Parameters
    ----------
    predictions : list[dict]
        Lista de previsoes (mix de resolvidas e pendentes).

    Returns
    -------
    dict
        Sumario com total, resolved, pending, accuracy_1x2,
        accuracy_over_under, avg_brier, by_round, by_team.
    """
    total = len(predictions)
    resolved = [p for p in predictions if p.get("status") == "resolved"]
    pending = [p for p in predictions if p.get("status") != "resolved"]

    n_resolved = len(resolved)
    n_pending = len(pending)

    if n_resolved == 0:
        return {
            "total": total,
            "resolved": 0,
            "pending": n_pending,
            "accuracy_1x2": 0.0,
            "accuracy_over_under": 0.0,
            "avg_brier": 0.0,
            "by_round": [],
            "by_team": {},
        }

    correct_1x2 = sum(1 for p in resolved if p.get("correct_1x2"))
    correct_ou = sum(1 for p in resolved if p.get("correct_over_under"))
    brier_scores = [p.get("brier_score", 0) or 0 for p in resolved]

    accuracy_1x2 = (correct_1x2 / n_resolved) * 100 if n_resolved else 0.0
    accuracy_over_under = (correct_ou / n_resolved) * 100 if n_resolved else 0.0
    avg_brier = sum(brier_scores) / n_resolved if n_resolved else 0.0

    # By round
    round_data: dict[int, dict] = defaultdict(lambda: {
        "total": 0, "correct_1x2": 0, "correct_ou": 0, "brier_sum": 0.0,
    })
    for p in resolved:
        r = p.get("round")
        if r is not None:
            round_data[r]["total"] += 1
            if p.get("correct_1x2"):
                round_data[r]["correct_1x2"] += 1
            if p.get("correct_over_under"):
                round_data[r]["correct_ou"] += 1
            round_data[r]["brier_sum"] += float(p.get("brier_score", 0) or 0)

    by_round = []
    for r_num in sorted(round_data.keys()):
        rd = round_data[r_num]
        by_round.append({
            "round": r_num,
            "total": rd["total"],
            "accuracy_1x2": (rd["correct_1x2"] / rd["total"] * 100) if rd["total"] else 0.0,
            "accuracy_ou": (rd["correct_ou"] / rd["total"] * 100) if rd["total"] else 0.0,
            "avg_brier": rd["brier_sum"] / rd["total"] if rd["total"] else 0.0,
        })

    # By team (home + away appearances)
    team_data: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "correct_1x2": 0,
    })
    for p in resolved:
        for team_key in ("home_team", "away_team"):
            team = p.get(team_key)
            if team:
                team_data[team]["total"] += 1
                if p.get("correct_1x2"):
                    team_data[team]["correct_1x2"] += 1

    by_team = {}
    for team, td in sorted(team_data.items()):
        by_team[team] = {
            "total": td["total"],
            "accuracy_1x2": (td["correct_1x2"] / td["total"] * 100) if td["total"] else 0.0,
        }

    return {
        "total": total,
        "resolved": n_resolved,
        "pending": n_pending,
        "accuracy_1x2": round(accuracy_1x2, 1),
        "accuracy_over_under": round(accuracy_over_under, 1),
        "avg_brier": round(avg_brier, 4),
        "by_round": by_round,
        "by_team": by_team,
    }
