---
tags:
  - pitch
  - calibration
  - isotonic
  - temperature-scaling
  - brier
  - ece
---

# Pitch — v1.11.0: Isotonic + Auto Method Selection

## Problem

The v1.10.0 model has **Brier 0.2437** on 78 resolved predictions (realistic target: 0.20, bookmaker line). The reliability diagram exposed systematic overconfidence:

| Confidence bin | n | Predicted | Actual | Gap |
|---|---|---|---|---|
| 33-40% | 6 | 37.9% | 16.7% | +21pts |
| 40-50% | 24 | 44.9% | 50.0% | -5pts ✓ |
| 50-60% | 21 | 55.1% | 38.1% | +17pts |
| 60-70% | 9 | 66.0% | 55.6% | +10pts |
| **70%+** | **18** | **81.1%** | **38.9%** | **+42pts** 🔥 |

The v1.9.0 Platt scaling (a=0.46, b=0.04) is timid: it compresses 81%→67% when it should drop to ~40%.

## Solution

Add 2 alternative calibration methods + auto-selection via CV Brier:

### 1. Isotonic Regression (non-parametric)
- Any monotonic distortion, doesn't assume sigmoid like Platt
- Needs n≥1000 to avoid overfitting
- Tends to dominate when there's enough data

### 2. Temperature Scaling (1-param, safe)
- `p_cal = p^(1/T) / Z` — single parameter
- Minimal overfitting risk
- Safe fallback for small n

### 3. Auto-selection
- Time-split 80/20 (last 20% is val)
- Fit 3 methods on train, evaluate Brier + ECE on val
- Choose the best, re-fit on the full dataset

## Architecture

**Affected modules:**
- `domain/calibration.py` — 2 new classes (`TemperatureScaler`, `IsotonicCalibrator`), 6 functions (fit + apply + metrics)
- `use_cases/fit_calibration.py` — refactor for multi-method CV
- `use_cases/predict_all.py` — dispatch by method in `_apply_calibration`
- `cli.py` — flag `--method auto|platt|isotonic|temperature`

**New pickle schema:**
```python
{
    "method": "platt" | "isotonic" | "temperature",
    "dixon_coles_rho": float,
    "platt_home/draw/away": {...},  # always saved (fallback)
    "iso_home/draw/away": {...},    # if isotonic
    "temperature": {"T": float},    # if temperature
    "cv_results": {method: {brier_val, ece_val} for 3 methods},
    "metrics": {"brier_raw", "brier_calibrated", "ece_raw", "ece_calibrated", ...}
}
```

## Scope

**In:**
- Isotonic + Temperature + ECE metrics
- Time-split CV + auto-select
- CLI flag
- Backward-compat dispatch (old pickle without `method` defaults to platt)
- 22 new tests

**Out:**
- Ingestion of 2022/2023 (obvious next step — attacks the root cause of low n)
- Beta calibration
- Hybrid temperature+isotonic residual

## Research

Based on `docs/research/calibration-methods.md`:
- Isotonic > Platt when n ≥ 1000 (Sports AI blog, scikit-learn docs)
- Temperature scaling is a post-hoc "millisecond fix" (Geoff Pleiss)
- **Calibration > Accuracy**: +34.69% ROI vs -35.17% (Wilkens, 2024)
- Bookmaker RPS benchmark: 0.17-0.22 (penaltyblog, 250M lines)
- The <0.19 target from pitch v1.6.0 is more ambitious than real bookmakers — adjust to 0.20

## Testing

Unit tests in `tests/test_domain_calibration.py`:
- TemperatureScaler: T=1 identity, T>1 compress, T<1 sharpen, normalization
- IsotonicCalibrator: identity, correction, monotonicity, empty thresholds
- fit_temperature: recovers T from synthetic data, reduces Brier on overconfident
- fit_isotonic_binary: monotonicity, Brier reduction
- compute_ece: perfect = 0, overconfident > 0.4
- compute_brier_3class: formula checks

## Success Criteria

1. **Functional**: `moneyball fit-calibration --method auto` shows CV comparison and selects winner
2. **Tests**: 295+ passing (22 new)
3. **Arch**: pure domain (only numpy/scipy/sklearn)
4. **Real metrics (409 samples)**: Platt wins CV (Brier_val 0.652 vs Iso 0.695 vs Temp 0.673)
5. **In-sample improvements**: Brier -3%, **ECE -53%** (0.060→0.028), Acc +1.4pts

## Result

**SHIPPED 2026-04-05** — full implementation, all tests passing.

**Empirical finding:** With n=409 still insufficient for isotonic to dominate (overfits on the val split). Auto-selection worked correctly by choosing Platt. Natural next pitch: **2022+2023 ingestion** to exceed n=1000 and unlock isotonic.

**Behavior on extreme input `[0.81, 0.10, 0.09]`:**
- Platt → 67.6%
- Isotonic → 83.0% (class-wise without joint constraint, inflated)
- Temperature → 63.5%

## Next linked pitches

- **v1.12.0 — Historical backfill 2022/2023** (~760 additional matches) → re-fit calibration, isotonic should dominate
- **v1.13.0 — Monitored calibration** (re-fit cron when ECE > 0.015)
- **v1.14.0 — Beta calibration / hybrid** (if isotonic alone doesn't reach target)
