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
    # Enriquecer com todos os mercados derivados + context (v1.6.0)
    for pred in predictions:
        pred["markets"] = derive_all_markets(pred)
        # v1.6.0: enrich com context (coach, injuries, standings)
        try:
            home = pred.get("home_team", "")
            away = pred.get("away_team", "")
            ct = pred.get("commence_time", "")
            if home and away and ct:
                pred["context"] = {
                    "home": {
                        "coach": repo.get_coach_change_info(home, ct),
                        "injuries": repo.get_key_players_out(home, ref_date=ct),
                    },
                    "away": {
                        "coach": repo.get_coach_change_info(away, ct),
                        "injuries": repo.get_key_players_out(away, ref_date=ct),
                    },
                    "standing": repo.get_standing_gap(home, away, ct),
                }
        except Exception:
            pred["context"] = None

    # Enriquecer com value bets associadas (deduplicadas, melhor odd por mercado)
    try:
        from football_moneyball.config import get_odds_provider
        from football_moneyball.use_cases.find_value_bets import FindValueBets
        odds_provider = get_odds_provider()
        odds_provider.repo = repo
        vb_result = FindValueBets(odds_provider, repo).execute(bankroll=1000, min_edge=0.05)
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
                    # 1X2: recomendar só se coerente com maior prob do modelo
                    # outcome pode ser "Home"/"Away"/"Draw" OU o nome do time
                    hp = pred.get("home_win_prob", 0)
                    dp = pred.get("draw_prob", 0)
                    ap = pred.get("away_win_prob", 0)
                    max_p = max(hp, dp, ap)
                    outcome = b["outcome"]
                    home_name = pred.get("home_team", "")
                    away_name = pred.get("away_team", "")
                    is_home_bet = outcome == "Home" or outcome == home_name
                    is_away_bet = outcome == "Away" or outcome == away_name
                    is_draw_bet = outcome == "Draw"
                    if is_draw_bet and dp == max_p:
                        coherent.append(b)
                    elif is_home_bet and hp == max_p:
                        coherent.append(b)
                    elif is_away_bet and ap == max_p:
                        coherent.append(b)
                elif b["market"] == "totals":
                    over25 = pred.get("over_25", 0.5)
                    if b["outcome"] == "Over" and over25 > 0.5:
                        coherent.append(b)
                    elif b["outcome"] == "Under" and over25 <= 0.5:
                        coherent.append(b)
                else:
                    coherent.append(b)

            # Traduzir outcome "Home"/"Away" → nome do time
            home_name = pred.get("home_team", "")
            away_name = pred.get("away_team", "")
            def _label(outcome: str) -> str:
                if outcome == "Over":
                    return "Mais de 2.5 gols"
                if outcome == "Under":
                    return "Menos de 2.5 gols"
                if outcome == "Draw":
                    return "Empate"
                if outcome == "Home":
                    return f"Vitória {home_name}"
                if outcome == "Away":
                    return f"Vitória {away_name}"
                if outcome in ("Yes",):
                    return "Ambas marcam"
                if outcome in ("No",):
                    return "Algum time não marca"
                return f"Vitória {outcome}"

            def _model_prob_for(outcome: str) -> float:
                """Traduz outcome pra probabilidade correspondente do modelo."""
                if outcome == "Over":
                    return float(pred.get("over_25", 0) or 0)
                if outcome == "Under":
                    return 1.0 - float(pred.get("over_25", 0) or 0)
                if outcome == "Draw":
                    return float(pred.get("draw_prob", 0) or 0)
                if outcome == "Home" or outcome == home_name:
                    return float(pred.get("home_win_prob", 0) or 0)
                if outcome == "Away" or outcome == away_name:
                    return float(pred.get("away_win_prob", 0) or 0)
                if outcome == "Yes":
                    return float(pred.get("btts_prob", 0) or 0)
                if outcome == "No":
                    return 1.0 - float(pred.get("btts_prob", 0) or 0)
                return 0.0

            pred["recommended_bets"] = [
                {
                    "market": b["market"],
                    "outcome": b["outcome"],
                    "odds": b["best_odds"],
                    "bookmaker": b["bookmaker"],
                    "edge": b["edge"],
                    "stake": b.get("stake", 0),
                    "model_prob": b.get("model_prob") or _model_prob_for(b["outcome"]),
                    "label": _label(b["outcome"]),
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
    """Retorna historico de previsoes com bets associadas (deduplicadas)."""
    preds = repo.get_prediction_history(round_num=round, status=status)
    bets = repo.get_value_bet_history()

    # Associar bets a predictions por match_key + dedup por market+outcome (melhor odd)
    bets_by_match: dict[int, list] = {}
    for b in bets:
        mk = b.get("match_key", 0)
        bets_by_match.setdefault(mk, []).append(b)

    # Build match_key → pred lookup pra traduzir Home/Away → nome do time
    pred_by_match: dict[int, dict] = {p.get("match_key", 0): p for p in preds}

    def _translate(bet: dict, pred: dict) -> dict:
        outcome = bet.get("outcome", "")
        if outcome == "Home":
            bet["outcome_label"] = f"Vitória {pred.get('home_team', '')}"
        elif outcome == "Away":
            bet["outcome_label"] = f"Vitória {pred.get('away_team', '')}"
        elif outcome == "Draw":
            bet["outcome_label"] = "Empate"
        elif outcome == "Over":
            bet["outcome_label"] = "Mais de 2.5 gols"
        elif outcome == "Under":
            bet["outcome_label"] = "Menos de 2.5 gols"
        else:
            bet["outcome_label"] = f"Vitória {outcome}" if outcome else ""
        return bet

    for p in preds:
        mk = p.get("match_key", 0)
        raw_bets = bets_by_match.get(mk, [])
        # Dedup: 1 por market+outcome (melhor odd, depois menor kelly_stake)
        seen: dict[str, dict] = {}
        for b in raw_bets:
            key = f"{b.get('market','')}-{b.get('outcome','')}"
            if key not in seen or (b.get("best_odds", 0) or 0) > (seen[key].get("best_odds", 0) or 0):
                seen[key] = b
        deduped = [_translate(b, p) for b in seen.values()]
        # Ordenar por edge desc
        deduped.sort(key=lambda x: -(x.get("edge", 0) or 0))
        p["bets"] = deduped

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


@app.get("/api/match-analysis/by-teams")
def get_match_analysis_by_teams(
    home_team: str, away_team: str, repo=Depends(get_repo),
):
    """Analise pos-jogo via nome dos times (fuzzy match)."""
    from sqlalchemy import text
    # Find match by team names (either direction — handle home/away swaps)
    row = repo._session.execute(text("""
        SELECT match_id FROM matches
        WHERE (home_team = :h AND away_team = :a)
           OR (home_team = :a AND away_team = :h)
           OR (REPLACE(home_team, 'São', 'Sao') = :h_norm AND REPLACE(away_team, 'São', 'Sao') = :a_norm)
           OR (REPLACE(home_team, 'São', 'Sao') = :a_norm AND REPLACE(away_team, 'São', 'Sao') = :h_norm)
        ORDER BY match_date DESC LIMIT 1
    """), {
        "h": home_team, "a": away_team,
        "h_norm": home_team.replace("São", "Sao").replace("é", "e").replace("í", "i"),
        "a_norm": away_team.replace("São", "Sao").replace("é", "e").replace("í", "i"),
    }).fetchone()
    if not row:
        return {"error": "Partida nao encontrada"}
    return get_match_analysis(int(row.match_id), repo)


@app.get("/api/match-analysis/{match_id}")
def get_match_analysis(match_id: int, repo=Depends(get_repo)):
    """Analise pos-jogo: previsoes vs stats reais de match_stats + prediction_history.

    v1.7.0 — compara o que o modelo previu com o que aconteceu.
    """
    from sqlalchemy import text

    # Match info
    match = repo._session.execute(text("""
        SELECT match_id, match_date, home_team, away_team, home_score, away_score, round
        FROM matches WHERE match_id = :mid
    """), {"mid": match_id}).fetchone()
    if not match or match.home_score is None:
        return {"error": "Partida nao encontrada ou sem resultado"}

    # Real match stats
    stats = repo._session.execute(text("""
        SELECT * FROM match_stats WHERE match_id = :mid
    """), {"mid": match_id}).fetchone()

    # Prediction that was made for this match (latest)
    pred = repo._session.execute(text("""
        SELECT * FROM prediction_history
        WHERE (home_team = :home AND away_team = :away)
           OR (home_team = :away AND away_team = :home)
        ORDER BY predicted_at DESC LIMIT 1
    """), {"home": match.home_team, "away": match.away_team}).fetchone()

    result = {
        "match": {
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "round": match.round,
            "match_date": str(match.match_date),
        },
        "real_stats": None,
        "prediction": None,
    }

    if stats:
        result["real_stats"] = {
            "possession": {"home": stats.home_possession, "away": stats.away_possession},
            "xg": {"home": stats.home_xg, "away": stats.away_xg},
            "shots": {"home": stats.home_shots, "away": stats.away_shots},
            "sot": {"home": stats.home_sot, "away": stats.away_sot},
            "saves": {"home": stats.home_saves, "away": stats.away_saves},
            "corners": {"home": stats.home_corners, "away": stats.away_corners},
            "cards": {"home": stats.home_yellow + (stats.home_red or 0),
                      "away": stats.away_yellow + (stats.away_red or 0)},
            "fouls": {"home": stats.home_fouls, "away": stats.away_fouls},
            "big_chances": {"home": stats.home_big_chances, "away": stats.away_big_chances},
            "big_chances_scored": {"home": stats.home_big_chances_scored, "away": stats.away_big_chances_scored},
            "touches_box": {"home": stats.home_touches_box, "away": stats.away_touches_box},
            "long_balls_pct": {"home": stats.home_long_balls_pct, "away": stats.away_long_balls_pct},
            "aerial_won_pct": {"home": stats.home_aerial_won_pct, "away": stats.away_aerial_won_pct},
            "goals_prevented": {"home": stats.home_goals_prevented, "away": stats.away_goals_prevented},
            "passes": {"home": stats.home_passes, "away": stats.away_passes},
            "pass_accuracy": {"home": stats.home_pass_accuracy, "away": stats.away_pass_accuracy},
            "dispossessed": {"home": stats.home_dispossessed, "away": stats.away_dispossessed},
            "ht_score": {"home": stats.ht_home_score, "away": stats.ht_away_score},
            "referee": stats.referee_name,
        }

    if pred:
        # Check if prediction is inverted vs actual match
        inverted = pred.home_team != match.home_team
        if inverted:
            hp, dp, ap = pred.away_win_prob, pred.draw_prob, pred.home_win_prob
        else:
            hp, dp, ap = pred.home_win_prob, pred.draw_prob, pred.away_win_prob

        result["prediction"] = {
            "home_win_prob": hp, "draw_prob": dp, "away_win_prob": ap,
            "home_xg_expected": pred.away_xg_expected if inverted else pred.home_xg_expected,
            "away_xg_expected": pred.home_xg_expected if inverted else pred.away_xg_expected,
            "most_likely_score": pred.most_likely_score,
            "over_25_prob": pred.over_25_prob,
            "btts_prob": pred.btts_prob,
            "brier_score": pred.brier_score,
            "correct_1x2": bool(pred.correct_1x2) if pred.correct_1x2 is not None else None,
            "inverted": inverted,
        }

    return result


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
