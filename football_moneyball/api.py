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
    """Retorna previsoes pre-computadas com interpretacao e bets recomendadas."""
    from football_moneyball.domain.markets import derive_all_markets
    predictions = repo.get_predictions()
    predictions = [_interpret_prediction(p) for p in predictions]
    # Enriquecer com todos os mercados derivados
    for pred in predictions:
        pred["markets"] = derive_all_markets(pred)

    # Enriquecer com value bets associadas (deduplicadas, melhor odd por mercado)
    try:
        from football_moneyball.config import get_odds_provider
        from football_moneyball.use_cases.find_value_bets import FindValueBets
        odds_provider = get_odds_provider()
        odds_provider.repo = repo
        vb_result = FindValueBets(odds_provider, repo).execute(bankroll=1000, min_edge=0.03)
        all_bets = vb_result.get("value_bets", [])

        # Filtrar Betfair only
        betfair_bets = [b for b in all_bets if 'betfair' in b.get("bookmaker", "").lower()]
        # Fallback: se Betfair não tem, usar melhor odd geral
        if not betfair_bets:
            betfair_bets = all_bets

        # Dedup: melhor edge por match+market (só 1 lado, não Over E Under)
        seen = {}
        for b in betfair_bets:
            key = f"{b.get('match','')}-{b.get('market','')}"
            if key not in seen or b.get("edge", 0) > seen[key].get("edge", 0):
                seen[key] = b
        deduped = list(seen.values())

        # Associar bets a predictions — SÓ bets coerentes com a previsão
        for pred in predictions:
            match_name = f"{pred.get('home_team','')} vs {pred.get('away_team','')}"
            match_bets = [b for b in deduped if b.get("match", "") == match_name]

            # Filtrar: só bets alinhadas com o que o modelo prevê
            coherent = []
            for b in match_bets:
                if b["market"] == "h2h":
                    # 1X2: recomendar só o favorito do modelo
                    hp = pred.get("home_win_prob", 0)
                    dp = pred.get("draw_prob", 0)
                    ap = pred.get("away_win_prob", 0)
                    max_p = max(hp, dp, ap)
                    if b["outcome"] == "Draw" and dp == max_p:
                        coherent.append(b)
                    elif b["outcome"] != "Draw" and hp == max_p and b["outcome"] != pred.get("away_team"):
                        coherent.append(b)
                    elif b["outcome"] != "Draw" and ap == max_p and b["outcome"] != pred.get("home_team"):
                        coherent.append(b)
                elif b["market"] == "totals":
                    over25 = pred.get("over_25", 0.5)
                    if b["outcome"] == "Over" and over25 > 0.5:
                        coherent.append(b)
                    elif b["outcome"] == "Under" and over25 <= 0.5:
                        coherent.append(b)
                else:
                    coherent.append(b)

            pred["recommended_bets"] = [
                {
                    "market": b["market"],
                    "outcome": b["outcome"],
                    "odds": b["best_odds"],
                    "bookmaker": b["bookmaker"],
                    "edge": b["edge"],
                    "stake": b.get("stake", 0),
                    "label": "Mais de 2.5 gols" if b["outcome"] == "Over" else
                             "Menos de 2.5 gols" if b["outcome"] == "Under" else
                             f"Vitória {b['outcome']}" if b["outcome"] != "Draw" else "Empate",
                }
                for b in coherent
            ]
    except Exception:
        pass

    # Adicionar sugestões do modelo pra TODAS as linhas (sem odds reais, só probabilidade)
    for pred in predictions:
        if "recommended_bets" not in pred:
            pred["recommended_bets"] = []

        markets = pred.get("markets", {})

        # Melhor aposta por mercado baseada na probabilidade do modelo
        suggestions = []

        # 1X2: favorito
        match_odds = markets.get("match_odds", [])
        if match_odds:
            best = max(match_odds, key=lambda x: x.get("prob", 0))
            if best["prob"] > 0.40:
                suggestions.append({
                    "label": best["outcome"],
                    "market": "match_odds",
                    "outcome": best["outcome"],
                    "model_prob": best["prob"],
                    "fair_odds": best["fair_odds"],
                    "source": "model",
                })

        # Over/Under: linha mais confiante
        for ou in markets.get("over_under", []):
            line = ou["line"]
            if ou["over_prob"] > 0.65:
                suggestions.append({
                    "label": f"Mais de {line} gols",
                    "market": "totals",
                    "outcome": "Over",
                    "model_prob": ou["over_prob"],
                    "fair_odds": ou["over_odds"],
                    "source": "model",
                })
            elif ou["under_prob"] > 0.65:
                suggestions.append({
                    "label": f"Menos de {line} gols",
                    "market": "totals",
                    "outcome": "Under",
                    "model_prob": ou["under_prob"],
                    "fair_odds": ou["under_odds"],
                    "source": "model",
                })

        # BTTS se confiante
        btts = markets.get("btts", {})
        if btts.get("yes_prob", 0) > 0.65:
            suggestions.append({"label": "Ambas marcam", "market": "btts", "outcome": "Yes", "model_prob": btts["yes_prob"], "fair_odds": btts["yes_odds"], "source": "model"})
        elif btts.get("no_prob", 0) > 0.65:
            suggestions.append({"label": "Algum time não marca", "market": "btts", "outcome": "No", "model_prob": btts["no_prob"], "fair_odds": btts["no_odds"], "source": "model"})

        # Correct score: top 1
        cs = markets.get("correct_score", [])
        if cs and cs[0]["prob"] > 0.10:
            suggestions.append({"label": f"Placar exato {cs[0]['score']}", "market": "correct_score", "outcome": cs[0]["score"], "model_prob": cs[0]["prob"], "fair_odds": cs[0]["fair_odds"], "source": "model"})

        # Mesclar: bets com edge real primeiro, depois sugestões do modelo
        existing_labels = {b["label"] for b in pred.get("recommended_bets", [])}
        for s in suggestions:
            if s["label"] not in existing_labels:
                pred["recommended_bets"].append({
                    "label": s["label"],
                    "market": s["market"],
                    "outcome": s["outcome"],
                    "odds": s["fair_odds"],
                    "bookmaker": "modelo",
                    "edge": None,
                    "stake": None,
                    "model_prob": s["model_prob"],
                    "source": "model",
                })

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
        from football_moneyball.config import get_odds_provider
        r = get_repository()
        try:
            odds = get_odds_provider()
            odds.repo = r
            PredictAll(r, odds_provider=odds).execute(competition, season)
        except Exception:
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
    bookmaker: str | None = "betfair",
    repo=Depends(get_repo),
):
    """Retorna value bets deduplicadas (Betfair-only por padrao)."""
    from football_moneyball.config import get_odds_provider
    from football_moneyball.use_cases.find_value_bets import FindValueBets
    try:
        odds_provider = get_odds_provider()
        odds_provider.repo = repo
        result = FindValueBets(odds_provider, repo).execute(
            bankroll=bankroll, min_edge=min_edge, bookmaker_filter=bookmaker,
        )

        bets = result.get("value_bets", [])

        # Deduplicar: 1 linha por match+market+outcome (melhor odd)
        seen = {}
        for b in bets:
            key = f"{b.get('match','')}-{b.get('market','')}-{b.get('outcome','')}"
            if key not in seen or b.get("best_odds", 0) > seen[key].get("best_odds", 0):
                seen[key] = b
        deduped = sorted(seen.values(), key=lambda x: -x.get("edge", 0))

        result["value_bets"] = deduped
        result["total_before_dedup"] = len(bets)
        result["bookmaker_filter"] = bookmaker or "all"
        return result
    except Exception as e:
        return {"error": str(e), "value_bets": []}


@app.get("/api/track-record")
def get_track_record(repo=Depends(get_repo)):
    """Resumo do track record."""
    from football_moneyball.domain.track_record import calculate_track_record
    preds = repo.get_prediction_history()
    return calculate_track_record(preds)


@app.get("/api/track-record/predictions")
def get_track_record_predictions(
    round: int | None = None,
    status: str | None = None,
    repo=Depends(get_repo),
):
    """Retorna historico de previsoes com bets associadas."""
    preds = repo.get_prediction_history(round_num=round, status=status)
    bets = repo.get_value_bet_history()

    # Associar bets a predictions por match_key
    bets_by_match: dict[int, list] = {}
    for b in bets:
        mk = b.get("match_key", 0)
        bets_by_match.setdefault(mk, []).append(b)

    for p in preds:
        mk = p.get("match_key", 0)
        p["bets"] = bets_by_match.get(mk, [])

    return preds


@app.get("/api/track-record/value-bets")
def get_track_record_value_bets(repo=Depends(get_repo)):
    """Retorna historico de value bets."""
    return repo.get_value_bet_history()


@app.post("/api/resolve")
def trigger_resolve(repo=Depends(get_repo)):
    """Resolve previsoes pendentes com resultados reais."""
    from football_moneyball.use_cases.resolve_predictions import ResolvePredictions
    return ResolvePredictions(repo).execute()


@app.get("/api/verify")
def get_verify(
    competition: str = "Brasileirão Série A",
    season: str = "2026",
    repo=Depends(get_repo),
):
    """Verifica previsoes vs resultados reais (de prediction_history resolvido)."""
    preds = repo.get_prediction_history(status="resolved")
    if not preds:
        return {
            "error": "Nenhuma previsao resolvida. Rode 'moneyball resolve' apos jogos terminarem.",
            "total_matches": 0,
        }

    predictions = []
    total = len(preds)
    correct_1x2 = 0
    correct_ou = 0
    brier_sum = 0.0

    for p in preds:
        hp = p.get("home_win_prob") or 0
        dp = p.get("draw_prob") or 0
        ap = p.get("away_win_prob") or 0
        actual = p.get("actual_outcome", "")
        home_goals = p.get("actual_home_goals") or 0
        away_goals = p.get("actual_away_goals") or 0

        probs = {"home": hp, "draw": dp, "away": ap}
        predicted = max(probs, key=probs.get)
        predicted_label = {
            "home": p.get("home_team", "?"),
            "draw": "Empate",
            "away": p.get("away_team", "?"),
        }[predicted]
        actual_label = {
            "home": p.get("home_team", "?"),
            "draw": "Empate",
            "away": p.get("away_team", "?"),
        }.get(actual, actual)

        if p.get("correct_1x2"):
            correct_1x2 += 1
        if p.get("correct_over_under"):
            correct_ou += 1
        brier_sum += float(p.get("brier_score") or 0)

        predictions.append({
            "match": f"{p.get('home_team')} vs {p.get('away_team')}",
            "score": f"{home_goals} x {away_goals}",
            "predicted": predicted_label,
            "actual": actual_label,
            "correct_1x2": bool(p.get("correct_1x2")),
            "home_prob": hp, "draw_prob": dp, "away_prob": ap,
            "correct_over": bool(p.get("correct_over_under")),
            "brier": p.get("brier_score"),
        })

    return {
        "total_matches": total,
        "correct_1x2": correct_1x2,
        "correct_over_under": correct_ou,
        "accuracy_1x2": round(correct_1x2 / total * 100, 1) if total else 0,
        "accuracy_over_under": round(correct_ou / total * 100, 1) if total else 0,
        "avg_brier_score": round(brier_sum / total, 4) if total else 0,
        "predictions": predictions,
    }


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
