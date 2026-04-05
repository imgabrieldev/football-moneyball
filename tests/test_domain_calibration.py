"""Testes para football_moneyball.domain.calibration."""

import numpy as np

from football_moneyball.domain.calibration import (
    IsotonicCalibrator,
    PlattParams,
    TemperatureScaler,
    bivariate_poisson_score_matrix,
    calibrate_1x2,
    calibrate_1x2_isotonic,
    calibrate_1x2_temperature,
    compute_brier_3class,
    compute_ece,
    dixon_coles_score_matrix,
    dixon_coles_tau,
    fit_dixon_coles_rho,
    fit_isotonic_binary,
    fit_lambda3,
    fit_platt_binary,
    fit_temperature,
    sample_scores_bivariate,
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


# ---------------------------------------------------------------------------
# v1.11.0: Temperature scaling tests
# ---------------------------------------------------------------------------

class TestTemperatureScaler:
    def test_T_equals_1_is_identity(self):
        t = TemperatureScaler(T=1.0)
        probs = np.array([[0.6, 0.25, 0.15]])
        out = t.apply(probs)
        assert np.allclose(out, probs, atol=1e-9)

    def test_T_greater_than_1_compresses_extremes(self):
        t = TemperatureScaler(T=3.0)
        probs = np.array([0.9, 0.07, 0.03])
        out = t.apply(probs)
        # Máximo deve cair, mínimo deve subir
        assert out[0] < 0.9
        assert out[2] > 0.03

    def test_T_less_than_1_sharpens(self):
        t = TemperatureScaler(T=0.3)
        probs = np.array([0.5, 0.3, 0.2])
        out = t.apply(probs)
        # Máximo deve subir
        assert out[0] > 0.5
        assert out[2] < 0.2

    def test_output_sums_to_one(self):
        t = TemperatureScaler(T=2.0)
        probs = np.random.default_rng(0).dirichlet([1, 1, 1], size=10)
        out = t.apply(probs)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


class TestFitTemperature:
    def test_reduces_nll_on_overconfident_data(self):
        # Overconfident: modelo prediz 81% mas real é 40%
        rng = np.random.default_rng(42)
        n = 400
        # Todas predições confiantes em home
        raw = np.tile([0.81, 0.10, 0.09], (n, 1))
        # Mas home só ganha 40% das vezes
        y = np.zeros((n, 3), dtype=int)
        for i in range(n):
            u = rng.uniform(0, 1)
            if u < 0.40:
                y[i, 0] = 1
            elif u < 0.65:
                y[i, 1] = 1
            else:
                y[i, 2] = 1

        brier_raw = compute_brier_3class(raw, y)
        t = fit_temperature(raw, y)
        # Temperature fittada deve ser > 1 (compressão pra reduzir overconfidence)
        assert t.T > 1.0
        cal = t.apply(raw)
        brier_cal = compute_brier_3class(cal, y)
        assert brier_cal < brier_raw

    def test_well_calibrated_data_T_near_1(self):
        rng = np.random.default_rng(7)
        n = 500
        probs = rng.dirichlet([2, 1, 2], size=n)
        # Labels drawn from the probs (well-calibrated)
        idx = np.array([rng.choice(3, p=probs[i]) for i in range(n)])
        y = np.zeros((n, 3), dtype=int)
        y[np.arange(n), idx] = 1
        t = fit_temperature(probs, y)
        # T should be close to 1 (within loose tolerance)
        assert 0.5 < t.T < 2.5


# ---------------------------------------------------------------------------
# v1.11.0: Isotonic tests
# ---------------------------------------------------------------------------

class TestIsotonicCalibrator:
    def test_identity_mapping(self):
        iso = IsotonicCalibrator(
            x_thresholds=[0.0, 0.5, 1.0],
            y_thresholds=[0.0, 0.5, 1.0],
        )
        assert abs(float(iso.apply(0.3)) - 0.3) < 1e-9
        assert abs(float(iso.apply(0.7)) - 0.7) < 1e-9

    def test_corrects_overconfidence_synthetic(self):
        # Map: raw 0.8 → cal 0.5 (compressão)
        iso = IsotonicCalibrator(
            x_thresholds=[0.0, 0.4, 0.8, 1.0],
            y_thresholds=[0.0, 0.3, 0.5, 0.6],
        )
        assert abs(float(iso.apply(0.8)) - 0.5) < 1e-9
        # Interpolação linear entre 0.4 e 0.8: @ 0.6 → 0.4
        assert abs(float(iso.apply(0.6)) - 0.4) < 1e-9

    def test_monotonic(self):
        iso = IsotonicCalibrator(
            x_thresholds=[0.0, 0.3, 0.6, 1.0],
            y_thresholds=[0.05, 0.2, 0.55, 0.9],
        )
        xs = np.linspace(0.0, 1.0, 50)
        ys = iso.apply(xs)
        assert np.all(np.diff(ys) >= -1e-12)

    def test_empty_thresholds_identity(self):
        iso = IsotonicCalibrator(x_thresholds=[], y_thresholds=[])
        assert abs(float(iso.apply(0.5)) - 0.5) < 1e-9


class TestFitIsotonicBinary:
    def test_monotonic_output(self):
        rng = np.random.default_rng(0)
        n = 200
        p = rng.uniform(0, 1, n)
        labels = (rng.uniform(0, 1, n) < p).astype(int)
        iso = fit_isotonic_binary(p, labels)
        ys = iso.apply(np.linspace(0, 1, 100))
        assert np.all(np.diff(ys) >= -1e-9)

    def test_reduces_brier_on_miscalibrated(self):
        # Raw overconfident: prediz p mas real é p^2
        rng = np.random.default_rng(1)
        n = 500
        p = rng.uniform(0.05, 0.95, n)
        labels = (rng.uniform(0, 1, n) < p ** 2).astype(int)
        iso = fit_isotonic_binary(p, labels)
        cal = iso.apply(p)
        brier_raw = float(np.mean((p - labels) ** 2))
        brier_cal = float(np.mean((cal - labels) ** 2))
        assert brier_cal < brier_raw


class TestCalibrate1x2Isotonic:
    def test_identity_with_noop_calibrators(self):
        iso_identity = IsotonicCalibrator(
            x_thresholds=[0.0, 1.0], y_thresholds=[0.0, 1.0],
        )
        raw = np.array([[0.5, 0.3, 0.2]])
        cal = calibrate_1x2_isotonic(raw, iso_identity, iso_identity, iso_identity)
        assert np.allclose(cal.sum(axis=1), 1.0, atol=1e-9)
        assert np.allclose(cal, raw, atol=1e-9)

    def test_renormalizes(self):
        # Calibradores que reduzem cada prob pela metade
        iso_half = IsotonicCalibrator(
            x_thresholds=[0.0, 1.0], y_thresholds=[0.0, 0.5],
        )
        raw = np.array([[0.5, 0.3, 0.2]])
        cal = calibrate_1x2_isotonic(raw, iso_half, iso_half, iso_half)
        # Mesmo comprimindo igualmente, renormalização devolve as proporções
        assert np.allclose(cal.sum(axis=1), 1.0, atol=1e-9)
        assert np.allclose(cal, raw, atol=1e-9)

    def test_1d_input(self):
        iso_identity = IsotonicCalibrator(
            x_thresholds=[0.0, 1.0], y_thresholds=[0.0, 1.0],
        )
        raw = np.array([0.5, 0.3, 0.2])
        cal = calibrate_1x2_isotonic(raw, iso_identity, iso_identity, iso_identity)
        assert cal.shape == (3,)
        assert abs(cal.sum() - 1.0) < 1e-9


class TestCalibrate1x2Temperature:
    def test_compresses_high_confidence(self):
        temp = TemperatureScaler(T=2.5)
        raw = np.array([[0.85, 0.10, 0.05]])
        cal = calibrate_1x2_temperature(raw, temp)
        assert cal[0, 0] < 0.85
        assert abs(cal.sum() - 1.0) < 1e-9

    def test_T_one_identity(self):
        temp = TemperatureScaler(T=1.0)
        raw = np.array([[0.5, 0.3, 0.2]])
        cal = calibrate_1x2_temperature(raw, temp)
        assert np.allclose(cal, raw, atol=1e-9)


# ---------------------------------------------------------------------------
# v1.11.0: Metrics tests
# ---------------------------------------------------------------------------

class TestComputeBrier3class:
    def test_perfect_prediction_zero(self):
        probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y = np.array([[1, 0, 0], [0, 1, 0]])
        assert compute_brier_3class(probs, y) == 0.0

    def test_uniform_reference(self):
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
        y = np.array([[1, 0, 0]])
        # (2/3)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9
        assert abs(compute_brier_3class(probs, y) - 6.0 / 9) < 1e-9


class TestComputeECE:
    def test_perfect_calibration_low_ece(self):
        rng = np.random.default_rng(0)
        n = 1000
        probs = rng.dirichlet([2, 2, 2], size=n)
        idx = np.array([rng.choice(3, p=probs[i]) for i in range(n)])
        y = np.zeros((n, 3), dtype=int)
        y[np.arange(n), idx] = 1
        ece = compute_ece(probs, y, n_bins=10)
        assert ece < 0.1

    def test_overconfident_high_ece(self):
        # Modelo sempre prevê 90% na classe 0, mas acerta só 40%
        n = 500
        probs = np.tile([0.9, 0.05, 0.05], (n, 1))
        rng = np.random.default_rng(42)
        y = np.zeros((n, 3), dtype=int)
        correct = rng.uniform(0, 1, n) < 0.4
        y[correct, 0] = 1
        y[~correct, 2] = 1
        ece = compute_ece(probs, y, n_bins=5)
        # 90% conf vs 40% acc → ~0.5 ECE
        assert ece > 0.4

    def test_empty_input_zero(self):
        probs = np.zeros((0, 3))
        y = np.zeros((0, 3))
        assert compute_ece(probs, y) == 0.0


# ---------------------------------------------------------------------------
# Bivariate Poisson diagonal-inflated (v1.12.0)
# ---------------------------------------------------------------------------

class TestBivariatePoissonScoreMatrix:
    def test_sums_to_one(self):
        m = bivariate_poisson_score_matrix(1.5, 1.0, lambda3=0.10)
        assert abs(m.sum() - 1.0) < 1e-9

    def test_draw_inflation_vs_independent(self):
        indep = bivariate_poisson_score_matrix(1.5, 1.0, lambda3=0.0)
        inflated = bivariate_poisson_score_matrix(1.5, 1.0, lambda3=0.15)
        draw_indep = sum(indep[i, i] for i in range(11))
        draw_inflated = sum(inflated[i, i] for i in range(11))
        assert draw_inflated > draw_indep

    def test_lambda3_zero_equals_independent_poisson(self):
        biv = bivariate_poisson_score_matrix(1.5, 1.0, lambda3=0.0)
        indep = dixon_coles_score_matrix(1.5, 1.0, rho=0.0)
        assert np.allclose(biv, indep, atol=1e-6)

    def test_shape(self):
        m = bivariate_poisson_score_matrix(1.5, 1.0, lambda3=0.10, max_goals=8)
        assert m.shape == (9, 9)

    def test_non_negative(self):
        m = bivariate_poisson_score_matrix(2.0, 0.5, lambda3=0.20)
        assert np.all(m >= 0)


class TestSampleScoresBivariate:
    def test_shape(self):
        h, a = sample_scores_bivariate(1.5, 1.0, lambda3=0.10, n_simulations=100, seed=0)
        assert h.shape == (100,)
        assert a.shape == (100,)

    def test_mean_near_lambda(self):
        h, a = sample_scores_bivariate(1.5, 1.0, lambda3=0.10, n_simulations=50_000, seed=42)
        assert abs(h.mean() - 1.5) < 0.05
        assert abs(a.mean() - 1.0) < 0.05

    def test_seed_reproducible(self):
        h1, a1 = sample_scores_bivariate(1.5, 1.0, lambda3=0.10, n_simulations=100, seed=7)
        h2, a2 = sample_scores_bivariate(1.5, 1.0, lambda3=0.10, n_simulations=100, seed=7)
        assert np.array_equal(h1, h2)
        assert np.array_equal(a1, a2)

    def test_more_draws_than_independent(self):
        h_biv, a_biv = sample_scores_bivariate(1.5, 1.0, lambda3=0.15, n_simulations=100_000, seed=0)
        h_ind, a_ind = sample_scores_bivariate(1.5, 1.0, lambda3=0.0, n_simulations=100_000, seed=0)
        draws_biv = (h_biv == a_biv).mean()
        draws_ind = (h_ind == a_ind).mean()
        assert draws_biv > draws_ind


class TestFitLambda3:
    def test_recovers_lambda3_from_synthetic(self):
        # Gerar dados com lambda3=0.12
        rng = np.random.default_rng(42)
        matches = []
        for _ in range(300):
            lh, la = rng.uniform(0.8, 2.0), rng.uniform(0.5, 1.5)
            l3_true = 0.12
            x1 = rng.poisson(max(lh - l3_true, 0.05))
            x2 = rng.poisson(max(la - l3_true, 0.05))
            x3 = rng.poisson(l3_true)
            matches.append((lh, la, int(x1 + x3), int(x2 + x3)))
        fitted = fit_lambda3(matches)
        assert 0.05 < fitted < 0.25  # within reasonable range

    def test_zero_draws_gives_small_lambda3(self):
        # Sem empates → lambda3 deve ser baixo
        matches = [(1.5, 1.0, 3, 0)] * 50 + [(1.5, 1.0, 0, 2)] * 50
        fitted = fit_lambda3(matches)
        assert fitted < 0.10
