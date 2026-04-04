"""Use case: verificar previsoes vs resultados reais.

Compara as previsoes do modelo com os resultados reais das partidas
ja disputadas para medir acuracia e calibracao.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_moneyball.domain.match_predictor import simulate_match, estimate_team_xg


class VerifyPredictions:
    """Compara previsoes com resultados reais.

    Carrega snapshots de odds salvos, roda o modelo para cada jogo,
    e compara com o resultado real do Sofascore.

    Parameters
    ----------
    repo : MatchRepository
        Repositorio com resultados reais.
    """

    def __init__(self, repo) -> None:
        self.repo = repo

    def execute(
        self,
        snapshots_dir: Path | None = None,
        competition: str = "Brasileirão Série A",
        season: str = "2026",
    ) -> dict[str, Any]:
        """Verifica previsoes contra resultados reais.

        Para cada partida que ja aconteceu:
        1. Roda o modelo com dados pre-jogo
        2. Compara previsao vs resultado
        3. Calcula metricas de acuracia

        Returns
        -------
        dict
            Metricas e detalhes de cada previsao.
        """
        if snapshots_dir is None:
            snapshots_dir = Path(__file__).resolve().parent.parent.parent / "data" / "snapshots"

        # Carregar odds dos snapshots
        snapshot_odds = self._load_snapshots(snapshots_dir)
        if not snapshot_odds:
            return {"error": "Nenhum snapshot de odds encontrado em data/snapshots/"}

        # Buscar resultados reais do banco
        all_metrics = self.repo.get_all_metrics(competition, season)
        if all_metrics.empty:
            return {"error": "Sem dados de resultados no banco."}

        # Resultados por time por partida
        match_results = (
            all_metrics.groupby(["match_id", "team"])
            .agg(goals=("goals", "sum"), xg=("xg", "sum"))
            .reset_index()
        )

        # Para cada jogo nas odds, verificar se ja aconteceu
        predictions = []
        for game in snapshot_odds:
            home = game.get("home_team", "")
            away = game.get("away_team", "")

            # Buscar resultado real
            result = self._find_result(match_results, home, away)
            if result is None:
                continue  # jogo ainda nao aconteceu

            home_goals, away_goals = result

            # Rodar modelo
            home_history = self._get_team_history(all_metrics, home)
            away_history = self._get_team_history(all_metrics, away)
            home_def = self._get_defensive_history(all_metrics, home)
            away_def = self._get_defensive_history(all_metrics, away)

            home_xg = estimate_team_xg(home_history, away_def, is_home=True)
            away_xg = estimate_team_xg(away_history, home_def, is_home=False)
            pred = simulate_match(home_xg, away_xg)

            # Resultado real
            if home_goals > away_goals:
                actual = "Home"
            elif home_goals < away_goals:
                actual = "Away"
            else:
                actual = "Draw"

            # Previsao do modelo (maior probabilidade)
            probs = {
                "Home": pred["home_win_prob"],
                "Draw": pred["draw_prob"],
                "Away": pred["away_win_prob"],
            }
            predicted = max(probs, key=probs.get)

            # Brier score
            actual_vec = [1.0 if actual == k else 0.0 for k in ["Home", "Draw", "Away"]]
            pred_vec = [probs["Home"], probs["Draw"], probs["Away"]]
            brier = sum((p - a) ** 2 for p, a in zip(pred_vec, actual_vec))

            # Over/Under 2.5
            total_goals = home_goals + away_goals
            pred_over = pred["over_25"] > 0.5
            actual_over = total_goals > 2.5

            # Odds info (melhor odd 1X2)
            best_odds = self._get_best_odds(game, actual)

            predictions.append({
                "match": f"{home} vs {away}",
                "home_team": home,
                "away_team": away,
                "score": f"{home_goals}x{away_goals}",
                "actual": actual,
                "predicted": predicted,
                "correct_1x2": predicted == actual,
                "home_prob": pred["home_win_prob"],
                "draw_prob": pred["draw_prob"],
                "away_prob": pred["away_win_prob"],
                "over_25_prob": pred["over_25"],
                "pred_over": pred_over,
                "actual_over": actual_over,
                "correct_over": pred_over == actual_over,
                "brier": round(brier, 4),
                "home_xg_expected": round(home_xg, 2),
                "away_xg_expected": round(away_xg, 2),
                "best_odds_winner": best_odds,
            })

        if not predictions:
            return {"error": "Nenhuma partida verificavel (jogos ainda nao aconteceram)."}

        # Metricas agregadas
        df = pd.DataFrame(predictions)
        n = len(df)
        correct_1x2 = df["correct_1x2"].sum()
        correct_over = df["correct_over"].sum()
        avg_brier = df["brier"].mean()

        return {
            "total_matches": n,
            "correct_1x2": int(correct_1x2),
            "accuracy_1x2": round(correct_1x2 / n * 100, 1),
            "correct_over_under": int(correct_over),
            "accuracy_over_under": round(correct_over / n * 100, 1),
            "avg_brier_score": round(avg_brier, 4),
            "predictions": predictions,
        }

    def _load_snapshots(self, snapshots_dir: Path) -> list[dict]:
        """Carrega odds do snapshot mais recente."""
        if not snapshots_dir.exists():
            return []

        files = sorted(snapshots_dir.glob("odds_upcoming_*.json"))
        if not files:
            return []

        # Usar o mais recente
        with open(files[-1]) as f:
            return json.load(f)

    def _find_result(
        self, match_results: pd.DataFrame, home: str, away: str
    ) -> tuple[int, int] | None:
        """Busca resultado real de uma partida por nomes dos times."""
        # Fuzzy match por substring
        for mid in match_results["match_id"].unique():
            match_data = match_results[match_results["match_id"] == mid]
            teams = match_data["team"].tolist()
            if len(teams) < 2:
                continue

            home_match = any(home.lower() in t.lower() or t.lower() in home.lower() for t in teams)
            away_match = any(away.lower() in t.lower() or t.lower() in away.lower() for t in teams)

            if home_match and away_match:
                goals = {}
                for _, row in match_data.iterrows():
                    goals[row["team"]] = int(row["goals"])

                # Identificar home/away
                for t in teams:
                    if home.lower() in t.lower() or t.lower() in home.lower():
                        home_g = goals.get(t, 0)
                    else:
                        away_g = goals.get(t, 0)
                return home_g, away_g

        return None

    def _get_team_history(self, all_metrics: pd.DataFrame, team: str) -> pd.DataFrame:
        """Historico de xG ofensivo do time."""
        team_data = all_metrics[all_metrics["team"].str.contains(team, case=False, na=False)]
        if team_data.empty:
            return pd.DataFrame({"xg": [1.2]})
        per_match = (
            team_data.groupby("match_id")
            .agg(xg=("xg", "sum"))
            .reset_index()
            .sort_values("match_id", ascending=False)
        )
        return per_match

    def _get_defensive_history(self, all_metrics: pd.DataFrame, team: str) -> pd.DataFrame:
        """Historico de xG sofrido pelo time."""
        team_matches = all_metrics[
            all_metrics["team"].str.contains(team, case=False, na=False)
        ]["match_id"].unique()

        records = []
        for mid in team_matches:
            match_data = all_metrics[all_metrics["match_id"] == mid]
            opp = match_data[~match_data["team"].str.contains(team, case=False, na=False)]
            if not opp.empty:
                records.append({"match_id": mid, "xg_against": opp["xg"].sum()})

        if not records:
            return pd.DataFrame({"xg_against": [1.2]})
        return pd.DataFrame(records).sort_values("match_id", ascending=False)

    def _get_best_odds(self, game: dict, actual_outcome: str) -> float | None:
        """Retorna a melhor odd disponivel para o resultado que aconteceu."""
        outcome_map = {"Home": game.get("home_team", ""), "Away": game.get("away_team", ""), "Draw": "Draw"}
        target = outcome_map.get(actual_outcome, "")

        best = 0.0
        for bm in game.get("bookmakers", []):
            for m in bm.get("markets", []):
                if m.get("market") == "h2h":
                    name = m.get("outcome", "")
                    if (actual_outcome == "Draw" and name == "Draw") or \
                       (actual_outcome == "Home" and target.lower() in name.lower()) or \
                       (actual_outcome == "Away" and target.lower() in name.lower()):
                        if m.get("odds", 0) > best:
                            best = m["odds"]
        return best if best > 0 else None
