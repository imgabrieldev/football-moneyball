"""Use case: backtesting do modelo de previsao com dados historicos."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_moneyball.domain.match_predictor import simulate_match, estimate_team_xg
from football_moneyball.domain.value_detector import find_value_bets, odds_to_implied_prob
from football_moneyball.domain.bankroll import calculate_stake


class Backtest:
    """Roda backtesting do modelo com partidas ja disputadas.

    Simula como o modelo teria performado apostando em partidas
    passadas, usando apenas dados disponiveis ate o momento de
    cada partida (sem lookahead bias).

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

        Para cada partida (em ordem cronologica):
        1. Usa apenas partidas anteriores para estimar xG
        2. Roda Monte Carlo para prever resultado
        3. Simula aposta com Kelly fracionario
        4. Compara com resultado real

        Parameters
        ----------
        competition, season : str
            Filtros.
        initial_bankroll : float
            Bankroll inicial.
        min_edge : float
            Edge minimo para apostar.
        kelly_fraction : float
            Fracao do Kelly.
        min_matches_history : int
            Minimo de partidas passadas antes de comecar a apostar.

        Returns
        -------
        dict
            Resultados do backtest: ROI, hit rate, Brier score, etc.
        """
        # Buscar todas as partidas da temporada
        all_metrics = self.repo.get_all_metrics(competition, season)
        if all_metrics.empty:
            return {"error": "Sem dados para backtesting."}

        # Agregar xG por time por partida
        match_teams = (
            all_metrics.groupby(["match_id", "team"])
            .agg(xg=("xg", "sum"), goals=("goals", "sum"))
            .reset_index()
        )

        # Obter lista de partidas únicas ordenadas por match_id (proxy cronológico)
        match_ids = sorted(match_teams["match_id"].unique())

        bankroll = initial_bankroll
        bets = []
        predictions = []
        bankroll_history = [initial_bankroll]

        for i, match_id in enumerate(match_ids):
            # Dados desta partida
            match_data = match_teams[match_teams["match_id"] == match_id]
            teams = match_data["team"].unique()
            if len(teams) < 2:
                continue

            home_team = teams[0]
            away_team = teams[1]
            home_goals = int(match_data[match_data["team"] == home_team]["goals"].iloc[0])
            away_goals = int(match_data[match_data["team"] == away_team]["goals"].iloc[0])

            # Histórico: apenas partidas ANTES desta
            prior_ids = [mid for mid in match_ids if mid < match_id]
            if len(prior_ids) < min_matches_history:
                continue

            prior_data = match_teams[match_teams["match_id"].isin(prior_ids)]

            # Histórico ofensivo de cada time
            home_history = (
                prior_data[prior_data["team"] == home_team]
                .sort_values("match_id", ascending=False)
            )
            away_history = (
                prior_data[prior_data["team"] == away_team]
                .sort_values("match_id", ascending=False)
            )

            if home_history.empty or away_history.empty:
                continue

            # Histórico defensivo (xG sofrido)
            home_def = self._get_defensive_history(prior_data, home_team, prior_ids)
            away_def = self._get_defensive_history(prior_data, away_team, prior_ids)

            # Estimar xG
            home_xg = estimate_team_xg(home_history, away_def, is_home=True)
            away_xg = estimate_team_xg(away_history, home_def, is_home=False)

            # Simular
            pred = simulate_match(home_xg, away_xg, n_simulations=10_000)

            # Resultado real
            if home_goals > away_goals:
                actual_outcome = "Home"
            elif home_goals < away_goals:
                actual_outcome = "Away"
            else:
                actual_outcome = "Draw"

            actual_total = home_goals + away_goals
            actual_btts = home_goals > 0 and away_goals > 0

            # Brier score component (para 1X2)
            actual_vec = [
                1.0 if actual_outcome == "Home" else 0.0,
                1.0 if actual_outcome == "Draw" else 0.0,
                1.0 if actual_outcome == "Away" else 0.0,
            ]
            pred_vec = [pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]]
            brier = sum((p - a) ** 2 for p, a in zip(pred_vec, actual_vec))

            predictions.append({
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_xg_expected": home_xg,
                "away_xg_expected": away_xg,
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

            # Simular apostas (usando odds sintéticas baseadas no mercado)
            # Gerar odds "fair" a partir do modelo + margem típica de 5%
            synthetic_odds = self._generate_synthetic_odds(pred, margin=0.10)
            value_bets_found = find_value_bets(pred, synthetic_odds, min_edge)

            for vb in value_bets_found:
                stake = calculate_stake(
                    bankroll, vb["model_prob"], vb["best_odds"],
                    kelly_fraction=kelly_fraction
                )
                if stake <= 0:
                    continue

                # Determinar se a aposta ganhou
                won = self._bet_won(vb, actual_outcome, actual_total, actual_btts)
                profit = stake * (vb["best_odds"] - 1) if won else -stake

                bankroll += profit
                bets.append({
                    "match_id": match_id,
                    "match": f"{home_team} vs {away_team}",
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

        # Calcular métricas
        if not bets:
            return {
                "error": "Nenhuma aposta simulada.",
                "predictions": predictions,
                "matches_analyzed": len(predictions),
            }

        bets_df = pd.DataFrame(bets)
        total_staked = bets_df["stake"].sum()
        total_return = total_staked + bets_df["profit"].sum()
        roi = (total_return - total_staked) / total_staked * 100 if total_staked > 0 else 0
        hit_rate = bets_df["won"].mean() * 100
        avg_brier = np.mean([p["brier"] for p in predictions]) if predictions else 0

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

    def _get_defensive_history(
        self, data: pd.DataFrame, team: str, match_ids: list
    ) -> pd.DataFrame:
        """Calcula xG sofrido pelo time em partidas anteriores."""
        records = []
        team_matches = data[data["team"] == team]["match_id"].unique()
        for mid in team_matches:
            match_data = data[data["match_id"] == mid]
            opp = match_data[match_data["team"] != team]
            if not opp.empty:
                records.append({"match_id": mid, "xg_against": opp["xg"].sum()})
        if not records:
            return pd.DataFrame({"xg_against": [1.2]})
        return pd.DataFrame(records).sort_values("match_id", ascending=False)

    def _generate_synthetic_odds(self, pred: dict, margin: float = 0.10) -> list[dict]:
        """Gera odds sinteticas simulando casas de apostas reais.

        Adiciona margem + noise aleatorio para simular ineficiencias
        de mercado. Casas reais nao precificam com a mesma probabilidade
        que nosso modelo — essa divergencia gera value bets.
        """
        rng = np.random.default_rng()

        def prob_to_odds(p: float) -> float:
            if p <= 0:
                return 50.0
            # Margem da casa + noise (simula divergencia com nosso modelo)
            noise = rng.normal(0, 0.04)  # ±4% de variacao
            p_adjusted = p * (1 + margin) + noise
            p_adjusted = max(min(p_adjusted, 0.95), 0.02)
            return round(1.0 / p_adjusted, 2)

        return [{
            "name": "synthetic",
            "markets": [
                {"market": "h2h", "outcome": "Home", "odds": prob_to_odds(pred["home_win_prob"]), "implied_prob": pred["home_win_prob"]},
                {"market": "h2h", "outcome": "Draw", "odds": prob_to_odds(pred["draw_prob"]), "implied_prob": pred["draw_prob"]},
                {"market": "h2h", "outcome": "Away", "odds": prob_to_odds(pred["away_win_prob"]), "implied_prob": pred["away_win_prob"]},
                {"market": "totals", "outcome": "Over", "odds": prob_to_odds(pred["over_25"]), "point": 2.5, "implied_prob": pred["over_25"]},
                {"market": "totals", "outcome": "Under", "odds": prob_to_odds(1 - pred["over_25"]), "point": 2.5, "implied_prob": 1 - pred["over_25"]},
                {"market": "btts", "outcome": "Yes", "odds": prob_to_odds(pred["btts_prob"]), "implied_prob": pred["btts_prob"]},
                {"market": "btts", "outcome": "No", "odds": prob_to_odds(1 - pred["btts_prob"]), "implied_prob": 1 - pred["btts_prob"]},
            ],
        }]

    def _bet_won(
        self, vb: dict, actual_outcome: str, actual_total: int, actual_btts: bool
    ) -> bool:
        """Determina se uma aposta foi vencedora."""
        market = vb["market"]
        outcome = vb["outcome"]

        if market == "h2h":
            return outcome == actual_outcome
        elif market == "totals":
            if outcome == "Over":
                return actual_total > 2.5
            else:
                return actual_total <= 2.5
        elif market == "btts":
            if outcome == "Yes":
                return actual_btts
            else:
                return not actual_btts
        return False
