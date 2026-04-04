"""FastAPI — API REST do Football Moneyball.

Endpoints read-only pra servir dados ao frontend.
"""

from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from football_moneyball.config import get_repository


app = FastAPI(
    title="Football Moneyball API",
    description="Football analytics & betting value finder",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_repo():
    """Dependency injection do repository."""
    repo = get_repository()
    try:
        yield repo
    finally:
        repo.close()


@app.get("/health")
def health():
    """Healthcheck."""
    return {"status": "ok", "version": "0.6.0"}


@app.get("/api/matches")
def list_matches(
    competition: str = "Brasileirão Série A",
    season: str = "2026",
    repo=Depends(get_repo),
):
    """Lista partidas da temporada."""
    from sqlalchemy import text
    result = repo._session.execute(text(
        "SELECT match_id, match_date, home_team, away_team, home_score, away_score "
        "FROM matches WHERE competition = :comp AND season = :season "
        "ORDER BY match_date DESC"
    ), {"comp": competition, "season": season})

    return [
        {
            "match_id": r.match_id,
            "match_date": str(r.match_date),
            "home_team": r.home_team,
            "away_team": r.away_team,
            "home_score": r.home_score,
            "away_score": r.away_score,
        }
        for r in result
    ]


def _interpret_prediction(pred: dict) -> dict:
    """Adiciona interpretacao textual a uma previsao."""
    home = pred.get("home_team", "?")
    away = pred.get("away_team", "?")
    hp = pred.get("home_win_prob", 0) or 0
    dp = pred.get("draw_prob", 0) or 0
    ap = pred.get("away_win_prob", 0) or 0

    # Favorito
    if hp > ap + 0.15:
        fav = home
        conf = "forte" if hp > 0.60 else "leve"
        pred["interpretation"] = f"{fav} {conf} favorito em casa"
    elif ap > hp + 0.15:
        fav = away
        conf = "forte" if ap > 0.60 else "leve"
        pred["interpretation"] = f"{fav} {conf} favorito fora"
    elif dp > 0.30:
        pred["interpretation"] = "Jogo equilibrado, empate provável"
    else:
        pred["interpretation"] = "Jogo equilibrado e aberto"

    # Confiança
    max_prob = max(hp, dp, ap)
    if max_prob > 0.65:
        pred["confidence"] = "alta"
    elif max_prob > 0.45:
        pred["confidence"] = "media"
    else:
        pred["confidence"] = "baixa"

    # Gols
    over = pred.get("over_25", 0) or 0
    if over > 0.65:
        pred["goals_hint"] = "Jogo com muitos gols esperado"
    elif over < 0.35:
        pred["goals_hint"] = "Jogo fechado, poucos gols"
    else:
        pred["goals_hint"] = ""

    return pred


@app.get("/api/predictions")
def get_predictions(repo=Depends(get_repo)):
    """Retorna previsoes pre-computadas com interpretacao."""
    predictions = repo.get_predictions()
    predictions = [_interpret_prediction(p) for p in predictions]
    return {"predictions": predictions, "total": len(predictions)}


@app.post("/api/predictions/recompute")
def recompute_predictions(
    competition: str = "Brasileirão Série A",
    season: str = "2026",
    repo=Depends(get_repo),
):
    """Recomputa todas as previsoes (pode demorar ~30s)."""
    from football_moneyball.use_cases.predict_all import PredictAll
    import threading

    def _run():
        r = get_repository()
        try:
            PredictAll(r).execute(competition, season)
        finally:
            r.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "computing", "message": "Previsoes sendo recomputadas em background. Atualize em ~30s."}


@app.get("/api/players")
def get_players(
    competition: str = "Brasileirão Série A",
    season: str = "2026",
    team: str | None = None,
    repo=Depends(get_repo),
):
    """Lista jogadores com metricas agregadas."""
    from sqlalchemy import text
    params = {"comp": competition, "season": season}
    where = "WHERE m.competition = :comp AND m.season = :season"
    if team:
        where += " AND pmm.team = :team"
        params["team"] = team

    result = repo._session.execute(text(f"""
        SELECT pmm.player_name, pmm.team,
               COUNT(DISTINCT pmm.match_id) as matches,
               ROUND(SUM(pmm.minutes_played)::numeric, 0) as minutes,
               SUM(pmm.goals) as goals,
               SUM(pmm.assists) as assists,
               ROUND(SUM(pmm.xg)::numeric, 2) as xg,
               ROUND(SUM(pmm.xa)::numeric, 2) as xa,
               SUM(pmm.shots) as shots,
               SUM(pmm.passes) as passes,
               SUM(pmm.tackles) as tackles
        FROM player_match_metrics pmm
        JOIN matches m ON m.match_id = pmm.match_id
        {where}
        GROUP BY pmm.player_name, pmm.team
        HAVING SUM(pmm.minutes_played) > 0
        ORDER BY SUM(pmm.xg) DESC
        LIMIT 100
    """), params)

    return [
        {
            "player_name": r.player_name,
            "team": r.team,
            "matches": int(r.matches),
            "minutes": int(r.minutes or 0),
            "goals": int(r.goals or 0),
            "assists": int(r.assists or 0),
            "xg": float(r.xg or 0),
            "xa": float(r.xa or 0),
            "shots": int(r.shots or 0),
            "passes": int(r.passes or 0),
            "tackles": int(r.tackles or 0),
        }
        for r in result
    ]


@app.get("/api/value-bets")
def get_value_bets(
    bankroll: float = 1000.0,
    min_edge: float = 0.03,
    bookmaker: str | None = None,
    repo=Depends(get_repo),
):
    """Retorna value bets deduplicadas (melhor odd por aposta)."""
    from football_moneyball.config import get_odds_provider
    from football_moneyball.use_cases.find_value_bets import FindValueBets
    try:
        odds_provider = get_odds_provider()
        odds_provider.repo = repo
        result = FindValueBets(odds_provider, repo).execute(bankroll=bankroll, min_edge=min_edge)

        bets = result.get("value_bets", [])

        # Filtrar por bookmaker se pedido
        if bookmaker:
            bets = [b for b in bets if bookmaker.lower() in b.get("bookmaker", "").lower()]

        # Deduplicar: 1 linha por match+market+outcome (melhor odd)
        seen = {}
        for b in bets:
            key = f"{b.get('match','')}-{b.get('market','')}-{b.get('outcome','')}"
            if key not in seen or b.get("best_odds", 0) > seen[key].get("best_odds", 0):
                seen[key] = b
        deduped = sorted(seen.values(), key=lambda x: -x.get("edge", 0))

        result["value_bets"] = deduped
        result["total_before_dedup"] = len(bets)
        return result
    except Exception as e:
        return {"error": str(e), "value_bets": []}


@app.get("/api/verify")
def get_verify(
    competition: str = "Brasileirão Série A",
    season: str = "2026",
    repo=Depends(get_repo),
):
    """Verifica previsoes vs resultados."""
    from football_moneyball.use_cases.verify_predictions import VerifyPredictions
    result = VerifyPredictions(repo).execute(competition=competition, season=season)
    return result


@app.get("/api/backtest")
def get_backtest(
    competition: str = "Brasileirão Série A",
    season: str = "2026",
    bankroll: float = 1000.0,
    repo=Depends(get_repo),
):
    """Roda backtesting."""
    from football_moneyball.use_cases.backtest import Backtest
    result = Backtest(repo).execute(competition=competition, season=season, initial_bankroll=bankroll)
    # Remove large lists for API response
    result.pop("bets", None)
    result.pop("predictions", None)
    result.pop("bankroll_history", None)
    return result
