"""Tests for football_moneyball.domain.multi_monte_carlo."""

from football_moneyball.domain.multi_monte_carlo import (
    derive_markets_from_sims,
    simulate_full_match,
)


class TestSimulateFullMatch:
    def test_returns_dataframe_with_n_rows(self):
        lambdas = {
            "home_goals": 1.5, "away_goals": 1.0,
            "home_corners": 5.0, "away_corners": 4.0,
            "home_cards": 2.0, "away_cards": 2.0,
            "home_shots": 12.0, "away_shots": 10.0,
            "home_ht_goals": 0.67, "away_ht_goals": 0.45,
        }
        df = simulate_full_match(lambdas, n_simulations=1000, seed=42)
        assert len(df) == 1000

    def test_all_columns_present(self):
        lambdas = {"home_goals": 1.0, "away_goals": 1.0, "home_corners": 5.0,
                   "away_corners": 5.0, "home_cards": 2.0, "away_cards": 2.0,
                   "home_shots": 10.0, "away_shots": 10.0,
                   "home_ht_goals": 0.5, "away_ht_goals": 0.5}
        df = simulate_full_match(lambdas, n_simulations=100, seed=42)
        expected = ["home_goals", "away_goals", "home_corners", "away_corners",
                    "total_corners", "home_cards", "away_cards", "total_cards",
                    "home_shots", "away_shots", "ht_home", "ht_away"]
        for col in expected:
            assert col in df.columns

    def test_mean_approximates_lambda(self):
        lambdas = {"home_goals": 2.0, "away_goals": 1.0, "home_corners": 6.0,
                   "away_corners": 4.0, "home_cards": 2.5, "away_cards": 2.5,
                   "home_shots": 14.0, "away_shots": 9.0,
                   "home_ht_goals": 0.9, "away_ht_goals": 0.45}
        df = simulate_full_match(lambdas, n_simulations=10_000, seed=42)
        assert abs(df.home_goals.mean() - 2.0) < 0.1
        assert abs(df.home_corners.mean() - 6.0) < 0.3
        assert abs(df.home_shots.mean() - 14.0) < 0.3

    def test_deterministic_with_seed(self):
        lambdas = {"home_goals": 1.5, "away_goals": 1.0, "home_corners": 5.0,
                   "away_corners": 4.0, "home_cards": 2.0, "away_cards": 2.0,
                   "home_shots": 12.0, "away_shots": 10.0,
                   "home_ht_goals": 0.67, "away_ht_goals": 0.45}
        df1 = simulate_full_match(lambdas, n_simulations=100, seed=42)
        df2 = simulate_full_match(lambdas, n_simulations=100, seed=42)
        assert (df1 == df2).all().all()


class TestDeriveMarketsFromSims:
    def _make_sims(self):
        lambdas = {"home_goals": 1.5, "away_goals": 1.0, "home_corners": 5.0,
                   "away_corners": 4.0, "home_cards": 2.0, "away_cards": 2.0,
                   "home_shots": 12.0, "away_shots": 10.0,
                   "home_ht_goals": 0.67, "away_ht_goals": 0.45}
        return simulate_full_match(lambdas, n_simulations=10_000, seed=42)

    def test_corners_returned(self):
        sims = self._make_sims()
        markets = derive_markets_from_sims(sims)
        assert "corners" in markets
        assert len(markets["corners"]) >= 4
        assert "line" in markets["corners"][0]
        assert "over_prob" in markets["corners"][0]

    def test_cards_returned(self):
        sims = self._make_sims()
        markets = derive_markets_from_sims(sims)
        assert "cards" in markets

    def test_ht_result_sums_to_one(self):
        sims = self._make_sims()
        markets = derive_markets_from_sims(sims)
        ht = markets["ht_result"]
        total = ht["home_prob"] + ht["draw_prob"] + ht["away_prob"]
        assert abs(total - 1.0) < 0.01

    def test_margin_of_victory_keys(self):
        sims = self._make_sims()
        markets = derive_markets_from_sims(sims)
        assert "margin_of_victory" in markets
        assert "home_by_4_plus" in markets["margin_of_victory"]

    def test_probabilities_reasonable(self):
        sims = self._make_sims()
        markets = derive_markets_from_sims(sims)
        # Home favored (1.5 vs 1.0) -> HT home_prob should be > away_prob
        assert markets["ht_result"]["home_prob"] > markets["ht_result"]["away_prob"]

    def test_empty_df(self):
        import pandas as pd
        markets = derive_markets_from_sims(pd.DataFrame())
        assert markets == {}
