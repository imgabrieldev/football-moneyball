"""Use case: busca de value bets na rodada."""

from __future__ import annotations

from typing import Any

from football_moneyball.domain.value_detector import find_value_bets
from football_moneyball.domain.bankroll import calculate_stake


class FindValueBets:
    """Busca value bets comparando modelo com odds de casas de apostas.

    Parameters
    ----------
    odds_provider : OddsProvider
        Provedor de odds.
    repo : MatchRepository
        Repositorio para buscar historico.
    """

    def __init__(self, odds_provider, repo) -> None:
        self.odds_provider = odds_provider
        self.repo = repo

    def execute(
        self,
        bankroll: float = 1000.0,
        min_edge: float = 0.03,
        markets: list[str] | None = None,
    ) -> dict[str, Any]:
        """Busca value bets nas proximas partidas.

        Parameters
        ----------
        bankroll : float
            Valor do bankroll para calcular stakes.
        min_edge : float
            Edge minimo (default 3%).
        markets : list[str], optional
            Mercados a considerar.

        Returns
        -------
        dict
            Chaves: 'value_bets', 'total_matches', 'matches_with_value'.
        """
        from football_moneyball.use_cases.predict_match import PredictMatch

        # Buscar odds das proximas partidas
        upcoming = self.odds_provider.get_upcoming_odds(markets=markets)

        if not upcoming:
            return {"value_bets": [], "total_matches": 0, "matches_with_value": 0}

        predictor = PredictMatch(self.repo)
        all_value_bets = []

        for game in upcoming:
            home = game.get("home_team", "")
            away = game.get("away_team", "")

            # Se nomes vazios, extrair dos outcomes h2h
            if not home or not away:
                team_names = set()
                for bm in game.get("bookmakers", []):
                    for m in bm.get("markets", []):
                        if m.get("market") == "h2h" and m.get("outcome") != "Draw":
                            team_names.add(m["outcome"])
                team_names = sorted(team_names)
                if len(team_names) >= 2:
                    home, away = team_names[0], team_names[1]

            game_id = hash(f"{home}-{away}")  # synthetic ID

            # Predict
            try:
                prediction = predictor.execute(game_id, home, away)
            except Exception:
                continue

            # Collect all bookmaker odds for this game
            all_bm_odds = []
            for bm in game.get("bookmakers", []):
                all_bm_odds.append(bm)

            # Find value
            vbets = find_value_bets(prediction, all_bm_odds, min_edge, markets)

            # Calculate stakes
            for vb in vbets:
                vb["stake"] = calculate_stake(
                    bankroll, vb["model_prob"], vb["best_odds"]
                )
                vb["home_team"] = home
                vb["away_team"] = away
                vb["match"] = f"{home} vs {away}"

            all_value_bets.extend(vbets)

        matches_with_value = len(set(vb["match"] for vb in all_value_bets))

        # v1.4.1 — Persist ao histórico (pra backtest ROI depois)
        try:
            if all_value_bets:
                self.repo.save_value_bet_history(all_value_bets)
        except Exception:
            pass

        return {
            "value_bets": sorted(all_value_bets, key=lambda x: -x["edge"]),
            "total_matches": len(upcoming),
            "matches_with_value": matches_with_value,
        }
