---
tags:
  - pitch
  - calibration
  - monitoring
  - drift
  - ece
  - cronjob
---

# Pitch — v1.16.0: Monitored Calibration

## Problem

After v1.11.0 ([[../postmortem/isotonic-calibration-v1.11|isotonic-calibration-v1.11]]) calibration is a static pickle — fitted once, used until someone runs `fit-calibration` again. In practice this degrades because:

1. **Match distribution shifts**: transfer window, coaching changes, injuries, end-of-season fatigue → CatBoost confidences drift, ECE rises.
2. **v1.11 baseline was n=409**. With the v1.12 backfill (1610 matches) and new features in v1.14/v1.15, a re-fitted pickle captures more signal — but only if someone runs the command.
3. **Invisible ECE between fits**: there is no continuous monitoring. The gap is only discovered when the user runs `backtest` or `fit-calibration`.

Result: stale calibration is used to decide real bets, and the degradation stays hidden until an accidental manual re-fit.

## Solution

Daily K8s CronJob that:

1. **Computes ECE_live** on predictions resolved in the last 14 days (uses [calibration.py:469](football_moneyball/domain/calibration.py#L469), already in place)
2. **Compares** to the baseline saved in the pickle metadata
3. **Triggers a re-fit** if ECE > threshold **AND** accuracy has dropped, with a 7-day cooldown
4. **Validates the new pickle** on a hold-out before promoting — automatic rollback if worse
5. **Archives** old pickles in `/data/models/history/` for manual rollback
6. **Logs** metrics in a new `calibration_health` table for history/dashboard

## Architecture

### New modules

| Module | Responsibility |
|---|---|
| `domain/calibration_monitor.py` | Pure functions: `detect_drift(ece_live, ece_baseline, acc_live, acc_baseline) → Decision`, `validate_new_calibration(old, new, holdout) → bool` |
| `use_cases/monitor_calibration.py` | Orchestrates: fetches resolved predictions, computes ECE, decides, triggers re-fit, validates, promotes or rolls back |
| `adapters/postgres_repository.py` | `get_recent_resolved_predictions(days)`, `save_calibration_health(record)` |

### Modified modules

| Module | Change |
|---|---|
| `domain/calibration.py` | `CalibrationBundle` gains fields `baseline_ece`, `baseline_acc`, `fitted_at`, `n_training_samples` |
| `use_cases/fit_calibration.py` | Save baseline metadata in the pickle |
| `cli.py` | Command `monitor-calibration [--dry-run]` |
| `k8s/` | New `CronJob: calibration-monitor` (schedule: `0 3 * * *`) |

### Schema

New table:

```sql
CREATE TABLE calibration_health (
    id SERIAL PRIMARY KEY,
    checked_at TIMESTAMP NOT NULL DEFAULT NOW(),
    ece_live FLOAT NOT NULL,
    ece_baseline FLOAT NOT NULL,
    accuracy_live FLOAT NOT NULL,
    accuracy_baseline FLOAT NOT NULL,
    n_samples INT NOT NULL,
    decision TEXT NOT NULL,  -- 'ok' | 'warn' | 'refit_triggered' | 'refit_promoted' | 'refit_rolledback' | 'cooldown'
    old_pickle_path TEXT,
    new_pickle_path TEXT,
    notes TEXT
);
CREATE INDEX idx_calib_health_checked ON calibration_health(checked_at DESC);
```

Both in [db.py](football_moneyball/adapters/orm.py) ORM **and** [k8s/configmap.yaml](k8s/configmap.yaml) init.sql (rule: schema sync).

### Decision rules

```python
# Pure function in domain/calibration_monitor.py
def detect_drift(
    ece_live: float,
    ece_baseline: float,
    acc_live: float,
    acc_baseline: float,
    n_samples: int,
    days_since_last_fit: int,
) -> Decision:
    # Guardrails
    if n_samples < 40:
        return Decision.INSUFFICIENT_SAMPLES
    if days_since_last_fit < 7:
        return Decision.COOLDOWN

    ece_absolute_breach = ece_live > 0.05
    ece_relative_breach = ece_live > max(2 * ece_baseline, 0.03)
    acc_dropped = (acc_baseline - acc_live) > 0.03  # 3pp

    if (ece_absolute_breach or ece_relative_breach) and acc_dropped:
        return Decision.TRIGGER_REFIT
    if ece_absolute_breach or ece_relative_breach:
        return Decision.WARN
    return Decision.OK
```

Hold-out validation after re-fit:

```python
def validate_new_calibration(
    old_pickle, new_pickle, holdout_probs, holdout_labels
) -> bool:
    ece_old = compute_ece(old_pickle.apply(holdout_probs), holdout_labels)
    ece_new = compute_ece(new_pickle.apply(holdout_probs), holdout_labels)
    # Accept if new is 10%+ better OR within 5% of the old one
    return ece_new <= 0.9 * ece_old or ece_new <= 1.05 * ece_old
```

## Scope

### In Scope

- [ ] `domain/calibration_monitor.py` with `Decision` enum and pure functions
- [ ] `CalibrationBundle` gains baseline metadata (`baseline_ece`, `baseline_acc`, `fitted_at`, `n_training_samples`)
- [ ] `fit_calibration` saves metadata in the pickle
- [ ] `monitor_calibration` use case (orchestrates detect → refit → validate → promote/rollback)
- [ ] `calibration_health` table in ORM + init.sql
- [ ] CLI `monitor-calibration [--dry-run]`
- [ ] K8s CronJob `calibration-monitor` (daily at 3h UTC)
- [ ] Pickle history in `/data/models/history/<timestamp>.pkl` (keep last 5)
- [ ] Unit tests: `detect_drift` across 12 scenarios (cooldown, insufficient, ok, warn, trigger, rollback)

### Out of Scope

- External alerts (Slack, email) — only log + table
- Frontend dashboard for calibration health — only SQL query
- Automatic CatBoost retraining (v1.17)
- A/B testing between calibrations — only hold-out validation

## Research

See [[../research/calibration-monitoring|calibration-monitoring]]:

- ECE + accuracy combined gate is more robust than ECE alone
- Absolute threshold (ECE > 0.05) + relative threshold (ECE > 2× baseline) cover gradual and sudden drift
- 7-day cooldown avoids feedback loops
- Hold-out validation before promoting prevents regressions
- Min n ≥ 40 avoids sample noise (Brasileirão: ~4 matchdays)

Historical context: v1.11.0 had `ece_calibrated = 0.028` on 409 samples. The `0.05` threshold represents ~1.8× baseline — room for normal fluctuation without spurious alerts.

## Testing

### Unit (pure domain — zero mocks)

```python
# tests/test_calibration_monitor.py
def test_detect_drift_cooldown_blocks_refit()
def test_detect_drift_insufficient_samples()
def test_detect_drift_ok_when_ece_stable()
def test_detect_drift_warn_when_ece_up_acc_stable()
def test_detect_drift_trigger_when_ece_and_acc_breach()
def test_detect_drift_relative_vs_absolute_breach()
def test_validate_new_calibration_accepts_improvement()
def test_validate_new_calibration_accepts_within_tolerance()
def test_validate_new_calibration_rejects_regression()
```

### Integration

```python
# tests/test_monitor_calibration_usecase.py
def test_monitor_dry_run_does_not_write_pickle()
def test_monitor_full_flow_refit_promote()
def test_monitor_full_flow_refit_rollback_on_regression()
```

### Manual

- Run `monitor-calibration --dry-run` on the current dataset → verify there is no drift (low ECE post-v1.14)
- Inject predictions with shifted confidences → verify the trigger fires
- Simulate a re-fit with a worse pickle → verify the rollback

## Success Criteria

- [ ] `monitor-calibration --dry-run` runs in < 30s on the full dataset
- [ ] 12+ unit tests passing
- [ ] Domain layer with zero infra deps (`calibration_monitor.py` only imports numpy + `domain.calibration`)
- [ ] K8s CronJob running, metrics in `calibration_health`
- [ ] Old pickle archived in `history/` before any re-fit
- [ ] Functional rollback: manual test with a worse pickle reverts automatically
- [ ] Zero regression in manual `fit-calibration` — backward compat

## Related upcoming pitches

- v1.17.0 — CatBoost hyperopt + SHAP pruning (tuning the base model, not the calibration)
- v1.18.0 (future) — External alerts (webhook) when a trigger or rollback fires
- v1.19.0 (future) — Frontend dashboard for calibration health timeline
