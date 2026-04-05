"""Testes para football_moneyball.domain.calibration."""

import numpy as np

from football_moneyball.domain.calibration import (
    PlattParams,
    calibrate_1x2,
    dixon_coles_score_matrix,
    dixon_coles_tau,
    fit_dixon_coles_rho,
    fit_platt_binary,
    sample_scores_dixon_coles,
)


class TestDixonColesTau:
    def test_rho_zero_returns_ones(self):
        tau = dixon_coles_tau(1.5, 1.2, rho=0.0)
        assert np.allclose(tau, np.ones((2, 2)))

    def test_negative_rho_inflates_draws(self):
        # ρ < 0 deve aumentar τ(0,0) e τ(1,1) (mais empates)
        tau = dixon_coles_tau(1.5, 1.2, rho=-0.1)
        assert tau[0, 0] > 1.0
        assert tau[1, 1] > 1.0
        # E reduzir τ(0,1), τ(1,0) (menos 1-0 e 0-1)
        assert tau[0, 1] < 1.0
        assert tau[1, 0] < 1.0

    def test_non_negative(self):
        # τ nunca deve ficar negativo mesmo com ρ extremo
        tau = dixon_coles_tau(3.0, 3.0, rho=-0.5)
        assert (tau >= 0).all()


class TestDixonColesScoreMatrix:
    def test_sum_to_one(self):
        matrix = dixon_coles_score_matrix(1.5, 1.2, rho=-0.1)
        assert abs(matrix.sum() - 1.0) < 1e-9

    def test_rho_zero_equals_poisson_product(self):
        from scipy.stats import poisson
        lam_h, lam_a = 1.5, 1.2
        home_pmf = poisson.pmf(np.arange(11), lam_h)
        away_pmf = poisson.pmf(np.arange(11), lam_a)
        expected = np.outer(home_pmf, away_pmf)
        expected /= expected.sum()

        matrix = dixon_coles_score_matrix(lam_h, lam_a, rho=0.0)
        assert np.allclose(matrix, expected, atol=1e-9)

    def test_negative_rho_more_draws(self):
        # Compara massa em placares empatados vs ρ=0
        m_poisson = dixon_coles_score_matrix(1.5, 1.2, rho=0.0)
        m_dc = dixon_coles_score_matrix(1.5, 1.2, rho=-0.15)
        draws_poisson = sum(m_poisson[i, i] for i in range(11))
        draws_dc = sum(m_dc[i, i] for i in range(11))
        assert draws_dc > draws_poisson


class TestFitDixonColesRho:
    def test_all_draws_gives_negative_rho(self):
        # Se todas partidas foram 1-1, ρ deve ser bem negativo
        matches = [(1.5, 1.5, 1, 1) for _ in range(30)]
        rho = fit_dixon_coles_rho(matches)
        assert rho < 0
        assert rho >= -0.25

    def test_no_draws_gives_positive_or_zero(self):
        # Se nunca foram empates, ρ tende a 0 ou positivo
        matches = [(1.5, 1.0, 2, 0) for _ in range(30)]
        rho = fit_dixon_coles_rho(matches)
        assert rho > -0.05  # próximo de 0 ou positivo


class TestSampleScoresDixonColes:
    def test_shape(self):
        h, a = sample_scores_dixon_coles(1.5, 1.2, rho=-0.1, n_simulations=1000, seed=42)
        assert h.shape == (1000,)
        assert a.shape == (1000,)

    def test_mean_matches_lambda(self):
        h, a = sample_scores_dixon_coles(1.5, 1.2, rho=-0.1, n_simulations=50_000, seed=42)
        # Marginal deve ser próxima (Dixon-Coles só mexe levemente)
        assert abs(h.mean() - 1.5) < 0.15
        assert abs(a.mean() - 1.2) < 0.15

    def test_seed_reproducible(self):
        h1, a1 = sample_scores_dixon_coles(1.3, 1.1, rho=-0.1, n_simulations=500, seed=7)
        h2, a2 = sample_scores_dixon_coles(1.3, 1.1, rho=-0.1, n_simulations=500, seed=7)
        assert np.array_equal(h1, h2)
        assert np.array_equal(a1, a2)


class TestFitPlattBinary:
    def test_well_calibrated_returns_identity_ish(self):
        # Quando labels batem com probs, Platt não precisa ajustar muito
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.1, 0.9, size=500)
        labels = (rng.uniform(0, 1, size=500) < probs).astype(int)
        params = fit_platt_binary(probs, labels)
        # a deve ser próximo de 1, b próximo de 0
        assert 0.3 < params.a < 3.0
        assert abs(params.b) < 2.0

    def test_overconfident_model_gets_reduced(self):
        # Modelo que diz 80% mas acerta só 50% → deve reduzir probs altas
        probs = np.concatenate([
            np.full(100, 0.8),  # 100 prevendo 80%
            np.full(100, 0.2),  # 100 prevendo 20%
        ])
        labels = np.concatenate([
            np.concatenate([np.ones(50), np.zeros(50)]),  # acertou só 50%
            np.concatenate([np.ones(50), np.zeros(50)]),  # acertou 50% (deveria ter sido 20%)
        ])
        params = fit_platt_binary(probs, labels)
        # Aplicado a 0.8 deve dar algo mais próximo de 0.5
        cal_08 = params.apply(0.8)
        assert cal_08 < 0.75


class TestCalibrate1x2:
    def test_identity_params_no_change(self):
        raw = np.array([[0.5, 0.3, 0.2]])
        identity = PlattParams(a=1.0, b=0.0)
        cal = calibrate_1x2(raw, identity, identity, identity)
        assert np.allclose(cal, raw, atol=1e-9)

    def test_renormalizes_to_one(self):
        raw = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
        p1 = PlattParams(a=0.8, b=-0.2)
        p2 = PlattParams(a=1.2, b=0.1)
        p3 = PlattParams(a=0.9, b=0.0)
        cal = calibrate_1x2(raw, p1, p2, p3)
        assert np.allclose(cal.sum(axis=1), 1.0, atol=1e-9)

    def test_1d_input(self):
        raw = np.array([0.5, 0.3, 0.2])
        identity = PlattParams(a=1.0, b=0.0)
        cal = calibrate_1x2(raw, identity, identity, identity)
        assert cal.shape == (3,)
        assert abs(cal.sum() - 1.0) < 1e-9


class TestPlattParamsApply:
    def test_identity_params(self):
        p = PlattParams(a=1.0, b=0.0)
        assert abs(p.apply(0.5) - 0.5) < 1e-9
        assert abs(p.apply(0.3) - 0.3) < 1e-9

    def test_array_input(self):
        p = PlattParams(a=1.0, b=0.0)
        result = p.apply(np.array([0.1, 0.5, 0.9]))
        assert result.shape == (3,)
        assert np.allclose(result, [0.1, 0.5, 0.9], atol=1e-9)
