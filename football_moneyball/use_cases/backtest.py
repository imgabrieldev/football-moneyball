"""Use case: backtesting do modelo de previsao com dados historicos.

v0.5.0 — usa predict_match() com parametros dinamicos.
Para cada partida, usa apenas dados ANTERIORES (sem lookahead).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_moneyball.domain.match_predictor import predict_match
from football_moneyball.domain.value_detector import find_value_bets
from football_moneyball.domain.bankroll import calculate_stake


class Backtest:
    """Backtesting do modelo com partidas ja disputadas.

    Parameters
    ----------
    repo : MatchRepository
        Repositorio com dados historicos.
    """

    def __init__(self, repo) -> None:
        self.repo = repo

    def execute(
        self,
        competition: str = "Brasileirão Série A",
        season: str = "2026",
        initial_bankroll: float = 1000.0,
        min_edge: float = 0.03,
        kelly_fraction: float = 0.25,
        min_matches_history: int = 3,
    ) -> dict[str, Any]:
        """Executa backtesting completo.

        Para cada partida em ordem cronologica:
        1. Filtra dados para match_id < atual (sem lookahead)
        2. Roda predict_match() com dados pre-jogo
        3. Simula apostas com Kelly fracionario
        4. Compara com resultado real

        Returns
        -------
        dict
            ROI, hit rate, Brier score, detalhes de cada aposta.
        """
        # Buscar todos os dados
        all_data = self.repo.get_all_match_data(competition, season)
        if all_data.empty:
            return {"error": "Sem dados para backtesting."}

        # Lista de match_ids unicos em ordem
        match_ids = sorted(all_data["match_id"].unique())

        # Resultados reais por partida
        match_results = {}
        for mid in match_ids:
            md = all_data[all_data["match_id"] == mid]
            teams = md["team"].tolist()
            if len(teams) < 2:
                continue
            home_row = md[md["is_home"] == True]  # noqa: E712
            away_row = md[md["is_home"] == False]  # noqa: E712
            if home_row.empty or away_row.empty:
                continue
            match_results[mid] = {
                "home": home_row["team"].iloc[0],
                "away": away_row["team"].iloc[0],
                "home_goals": int(home_row["goals"].iloc[0]),
                "away_goals": int(away_row["goals"].iloc[0]),
            }

        bankroll = initial_bankroll
        bets = []
        predictions = []
        bankroll_history = [initial_bankroll]

        for i, mid in enumerate(match_ids):
            if mid not in match_results:
                continue

            result = match_results[mid]
            home = result["home"]
            away = result["away"]
            home_goals = result["home_goals"]
            away_goals = result["away_goals"]

            # Dados PRE-JOGO: apenas partidas anteriores
            prior_data = all_data[all_data["match_id"] < mid]
            n_prior_matches = len(prior_data["match_id"].unique())

            if n_prior_matches < min_matches_history * 10:  # ~3 rodadas completas
                continue

            # Verificar que ambos os times tem historico
            home_prior = prior_data[prior_data["team"] == home]
            away_prior = prior_data[prior_data["team"] == away]
            if home_prior.empty or away_prior.empty:
                continue

            # Prever com pipeline v0.5.0
            pred = predict_match(
                home_team=home,
                away_team=away,
                all_match_data=prior_data,
                n_simulations=10_000,
                seed=mid,  # reprodutivel
            )

            # Resultado real
            if home_goals > away_goals:
                actual_outcome = "Home"
            elif home_goals < away_goals:
                actual_outcome = "Away"
            else:
                actual_outcome = "Draw"

            total_goals = home_goals + away_goals
            actual_btts = home_goals > 0 and away_goals > 0

            # Brier score
            actual_vec = [
                1.0 if actual_outcome == "Home" else 0.0,
                1.0 if actual_outcome == "Draw" else 0.0,
                1.0 if actual_outcome == "Away" else 0.0,
            ]
            pred_vec = [pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]]
            brier = sum((p - a) ** 2 for p, a in zip(pred_vec, actual_vec))

            predictions.append({
                "match_id": mid,
                "home_team": home,
                "away_team": away,
                "home_xg_expected": pred["home_xg"],
                "away_xg_expected": pred["away_xg"],
                "home_win_prob": pred["home_win_prob"],
                "draw_prob": pred["draw_prob"],
                "away_win_prob": pred["away_win_prob"],
                "over_25_prob": pred["over_25"],
                "btts_prob": pred["btts_prob"],
                "actual_home_goals": home_goals,
                "actual_away_goals": away_goals,
                "actual_outcome": actual_outcome,
                "brier": brier,
            })

            # Simular apostas com odds sinteticas (margem 10% + noise)
            synthetic_odds = self._generate_synthetic_odds(pred, margin=0.10)
            value_bets_found = find_value_bets(pred, synthetic_odds, min_edge)

            for vb in value_bets_found:
                stake = calculate_stake(
                    bankroll, vb["model_prob"], vb["best_odds"],
                    kelly_fraction=kelly_fraction,
                )
                if stake <= 0:
                    continue

                won = self._bet_won(vb, actual_outcome, total_goals, actual_btts)
                profit = stake * (vb["best_odds"] - 1) if won else -stake
                bankroll += profit

                bets.append({
                    "match_id": mid,
                    "match": f"{home} vs {away}",
                    "market": vb["market"],
                    "outcome": vb["outcome"],
                    "model_prob": vb["model_prob"],
                    "odds": vb["best_odds"],
                    "edge": vb["edge"],
                    "stake": stake,
                    "won": won,
                    "profit": profit,
                    "bankroll_after": bankroll,
                })
                bankroll_history.append(bankroll)

        if not predictions:
            return {"error": "Nenhuma partida analisada.", "matches_analyzed": 0}

        if not bets:
            return {
                "error": "Nenhuma aposta simulada.",
                "predictions": predictions,
                "matches_analyzed": len(predictions),
                "avg_brier": round(np.mean([p["brier"] for p in predictions]), 4),
            }

        bets_df = pd.DataFrame(bets)
        total_staked = bets_df["stake"].sum()
        total_return = total_staked + bets_df["profit"].sum()
        roi = (total_return - total_staked) / total_staked * 100 if total_staked > 0 else 0
        hit_rate = bets_df["won"].mean() * 100
        avg_brier = np.mean([p["brier"] for p in predictions])

        # Max drawdown
        peak = initial_bankroll
        max_dd = 0
        for b in bankroll_history:
            if b > peak:
                peak = b
            dd = (peak - b) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": round(bankroll, 2),
            "total_staked": round(total_staked, 2),
            "total_return": round(total_return, 2),
            "roi": round(roi, 2),
            "hit_rate": round(hit_rate, 1),
            "brier_score": round(avg_brier, 4),
            "max_drawdown": round(max_dd * 100, 1),
            "matches_analyzed": len(predictions),
            "bets_placed": len(bets),
            "bets_won": int(bets_df["won"].sum()),
            "avg_edge": round(bets_df["edge"].mean() * 100, 2),
            "avg_odds": round(bets_df["odds"].mean(), 2),
            "bets": bets,
            "predictions": predictions,
            "bankroll_history": bankroll_history,
        }

    def _generate_synthetic_odds(self, pred: dict, margin: float = 0.10) -> list[dict]:
        """Odds sinteticas com margem + noise."""
        rng = np.random.default_rng()

        def prob_to_odds(p: float) -> float:
            if p <= 0:
                return 50.0
            noise = rng.normal(0, 0.04)
            p_adj = p * (1 + margin) + noise
            p_adj = max(min(p_adj, 0.95), 0.02)
            return round(1.0 / p_adj, 2)

        return [{
            "name": "synthetic",
            "markets": [
                {"market": "h2h", "outcome": "Home", "odds": prob_to_odds(pred["home_win_prob"])},
                {"market": "h2h", "outcome": "Draw", "odds": prob_to_odds(pred["draw_prob"])},
                {"market": "h2h", "outcome": "Away", "odds": prob_to_odds(pred["away_win_prob"])},
                {"market": "totals", "outcome": "Over", "odds": prob_to_odds(pred["over_25"]), "point": 2.5},
                {"market": "totals", "outcome": "Under", "odds": prob_to_odds(1 - pred["over_25"]), "point": 2.5},
            ],
        }]

    def _bet_won(self, vb: dict, actual: str, total: int, btts: bool) -> bool:
        """Determina se uma aposta foi vencedora."""
        if vb["market"] == "h2h":
            return vb["outcome"] == actual
        elif vb["market"] == "totals":
            return (total > 2.5) if vb["outcome"] == "Over" else (total <= 2.5)
        elif vb["market"] == "btts":
            return btts if vb["outcome"] == "Yes" else not btts
        return False
