---
tags:
  - research
  - calibration
  - monitoring
  - drift
  - ece
  - mlops
---

# Research — Automatic Calibration Monitoring and Retraining

> Research date: 2026-04-09
> Sources: listed at the end

## Context

In v1.11.0 ([[isotonic-calibration-v1.11]]) auto-selection between Platt / Isotonic / Temperature scaling with a pickle snapshot was implemented. The fit is manual — `moneyball fit-calibration` runs on demand. In production this model degrades: the match distribution changes (coach swaps, transfer windows, injuries, end-of-season fatigue), and the calibration becomes stale.

Goal: define **when** and **how** to trigger an automatic re-fit, and what guardrails to apply to avoid re-calibrating on noise.

## Findings

### 1. ECE as a monitoring metric

ECE (Expected Calibration Error) is the canonical metric to detect calibration drift when labels are available with delay — exactly the football case, where the real result comes out ~2h after the prediction.

**Formula** (already implemented in [calibration.py:469](football_moneyball/domain/calibration.py#L469)):

```
ECE = Σ_b (|B_b|/n) · |acc_b − conf_b|
```

Confidence bins (10 by default), weighted average of the gap between mean confidence and empirical accuracy.

### 2. Thresholds in the literature

There is no universal value. The recommended practice ([Medium: Angela Shi](https://medium.com/data-science-collective/a-practical-guide-to-prediction-drift-concept-drift-and-model-specific-monitoring-ee1ef086570b), [alldaystech](https://alldaystech.com/guides/artificial-intelligence/model-drift-detection-monitoring-response)) is:

1. **Establish a baseline**: ECE at fit time (training-set ECE after calibration). In v1.11.0: ECE calibrated = 0.028.
2. **Define an absolute threshold**: `ECE_limit = baseline + δ` where `δ ≈ baseline` (doubling rule).
3. **Define a relative threshold**: `ECE_live > 1.5× baseline` sustained.
4. **Minimum sample size**: do not alert with n < 30. Football: ~10 matches/matchday → minimum of 3 matchdays.
5. **Sustained window**: drift must last ≥ 2 consecutive measurements, not an isolated spike.

**Proposal for Moneyball:** ECE > 0.05 OR ECE > 2× baseline, sustained over 2 windows of 20 resolved predictions.

### 3. Accuracy-gated retrain

Calibration drift WITHOUT accuracy drift is dangerous — it may just be sampling noise. The recommendation ([alldaystech](https://alldaystech.com/guides/artificial-intelligence/model-drift-detection-monitoring-response)) is to combine:

- ECE > threshold **AND**
- Accuracy drop > 3pp vs 30-day baseline

Both must hit. Avoids re-calibrating on 20 atypical matches.

### 4. Cooldown and governance

Common problems when automating retraining without guardrails:

- **Feedback loops**: calibration adjusts to noise → worse ECE → re-adjusts. Minimum cooldown of 7 days between re-fits.
- **Parameter churn**: pickle changes every week, backtest becomes incomparable. Always snapshot with a timestamp.
- **Rollback**: if the new fit worsens out-of-sample ECE in the next window, revert to the previous pickle.

### 5. Online isotonic

Isotonic regression can be re-fit incrementally (scikit-learn `IsotonicRegression.fit` is O(n log n), fast enough to run a full refit in seconds for the ~500 accumulated samples). No sophisticated online learning required — re-fit from scratch inside a cronjob is enough.

### 6. Temperature scaling drift

Temperature scaling has 1 parameter (T). Direct monitoring: if `T_new / T_old > 1.3` or `< 0.77`, probabilities are being significantly compressed/stretched → sign of a real shift in the confidence distribution.

## Proposed architecture

### Flow

```
CronJob every 24h
  │
  ├─ 1. Fetch resolved predictions from the last 14 days (ResolvedPrediction)
  │     with calibration applied at prediction time
  │
  ├─ 2. Compute ECE_live, Brier_live, accuracy_live
  │
  ├─ 3. Compare vs baseline (stored in calibration pickle metadata)
  │
  ├─ 4. Decision:
  │     - ECE_live < threshold → OK, log metrics
  │     - ECE_live > threshold BUT accuracy stable → log warning, no action
  │     - ECE_live > threshold AND accuracy drop → TRIGGER re-fit
  │     - Cooldown active (< 7 days since last fit) → skip
  │
  ├─ 5. If triggered:
  │     - Run fit_calibration() on the full dataset
  │     - Compare new ECE vs old on hold-out (last 20%)
  │     - If better → save new pickle, archive the old one in /data/models/history/
  │     - If worse → alert, keep the old one
  │
  └─ 6. Write metrics into a calibration_health table
```

### Guardrails

- Min n: 40 resolved predictions since the last re-fit
- Cooldown: 7 days minimum
- Hold-out validation: the new pickle must improve ECE by ≥ 10% OR stay within 5% of the old one
- History: keep the last 5 pickles archived for rollback
- Alerts: if 3 consecutive triggers fail (new worse than old), pause automation and log an incident

## Conclusions

1. **ECE + accuracy combined gate** is the most robust rule — avoids re-calibrating on noise
2. **Absolute + relative thresholds** catch both gradual and sudden drift
3. **Cooldown** is mandatory to avoid feedback loops
4. **Hold-out validation** of the new pickle is cheap and prevents regressions
5. **History + rollback** provides a safety net

## Sources

- [A practical guide to prediction drift, concept drift, and model-specific monitoring — Angela Shi (Medium)](https://medium.com/data-science-collective/a-practical-guide-to-prediction-drift-concept-drift-and-model-specific-monitoring-ee1ef086570b)
- [Model Drift in Production: Detection, Monitoring & Response Runbook — AllDaysTech](https://alldaystech.com/guides/artificial-intelligence/model-drift-detection-monitoring-response)
- [Data Drift: Key Detection and Monitoring Techniques in 2026 — Label Your Data](https://labelyourdata.com/articles/machine-learning/data-drift)
- [Detect, Retrain, Repeat: Building a Model Drift Monitor with Alibi-Detect — Medium](https://medium.com/@myakalarajkumar1998/detect-retrain-repeat-building-a-model-drift-monitor-with-alibi-detect-ce1a32fe6cc5)
- [Classifier Calibration with ROC-Regularized Isotonic Regression — Berta et al., PMLR 2024](https://proceedings.mlr.press/v238/berta24a/berta24a.pdf)
- [[calibration-methods]] — prior research on Platt/Isotonic/Temperature
