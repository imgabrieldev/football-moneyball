---
tags:
  - pitch
  - calibration
  - monitoring
  - drift
  - ece
  - cronjob
---

# Pitch — v1.16.0: Calibração Monitorada

## Problema

Após v1.11.0 ([[../postmortem/isotonic-calibration-v1.11|isotonic-calibration-v1.11]]) a calibração é um pickle estático — fittado uma vez, usado até alguém rodar `fit-calibration` de novo. Na prática isso degrada porque:

1. **Distribuição de jogos muda**: janela de transferências, troca de técnicos, lesões, fadiga de fim de temporada → confidences do CatBoost deslocam, ECE sobe.
2. **Baseline v1.11 era n=409**. Com o backfill de v1.12 (1610 matches) e novas features de v1.14/v1.15, o pickle refittado captura mais sinal — mas só se alguém rodar o comando.
3. **ECE invisível entre fits**: não há monitoramento contínuo. Só quando o usuário roda `backtest` ou `fit-calibration` é que se descobre o gap.

Resultado: calibração estale é usada pra decidir apostas reais, e o degrade fica escondido até um re-fit manual acidental.

## Solução

CronJob diário no K8s que:

1. **Computa ECE_live** em predições resolvidas dos últimos 14 dias (usa [calibration.py:469](football_moneyball/domain/calibration.py#L469), já existente)
2. **Compara** com baseline salvo no metadata do pickle
3. **Dispara re-fit** se ECE > threshold **E** accuracy dropped, com cooldown de 7 dias
4. **Valida novo pickle** em hold-out antes de promover — rollback automático se pior
5. **Arquiva** pickles antigos em `/data/models/history/` pra rollback manual
6. **Loga** métricas em tabela nova `calibration_health` pra histórico/dashboard

## Arquitetura

### Novos módulos

| Módulo | Responsabilidade |
|---|---|
| `domain/calibration_monitor.py` | Funções puras: `detect_drift(ece_live, ece_baseline, acc_live, acc_baseline) → Decision`, `validate_new_calibration(old, new, holdout) → bool` |
| `use_cases/monitor_calibration.py` | Orquestra: pega predições resolvidas, computa ECE, decide, dispara re-fit, valida, promove ou rollback |
| `adapters/postgres_repository.py` | `get_recent_resolved_predictions(days)`, `save_calibration_health(record)` |

### Módulos modificados

| Módulo | Mudança |
|---|---|
| `domain/calibration.py` | `CalibrationBundle` ganha campos `baseline_ece`, `baseline_acc`, `fitted_at`, `n_training_samples` |
| `use_cases/fit_calibration.py` | Salvar baseline metadata no pickle |
| `cli.py` | Comando `monitor-calibration [--dry-run]` |
| `k8s/` | Novo `CronJob: calibration-monitor` (schedule: `0 3 * * *`) |

### Schema

Tabela nova:

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

Ambos em [db.py](football_moneyball/adapters/orm.py) ORM **e** [k8s/configmap.yaml](k8s/configmap.yaml) init.sql (rule: schema sync).

### Regras de decisão

```python
# Pure function em domain/calibration_monitor.py
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

Hold-out validation após re-fit:

```python
def validate_new_calibration(
    old_pickle, new_pickle, holdout_probs, holdout_labels
) -> bool:
    ece_old = compute_ece(old_pickle.apply(holdout_probs), holdout_labels)
    ece_new = compute_ece(new_pickle.apply(holdout_probs), holdout_labels)
    # Aceita se novo for 10%+ melhor OU within 5% do antigo
    return ece_new <= 0.9 * ece_old or ece_new <= 1.05 * ece_old
```

## Scope

### Dentro do Escopo

- [ ] `domain/calibration_monitor.py` com `Decision` enum e funções puras
- [ ] `CalibrationBundle` ganha metadata de baseline (`baseline_ece`, `baseline_acc`, `fitted_at`, `n_training_samples`)
- [ ] `fit_calibration` salva metadata no pickle
- [ ] `monitor_calibration` use case (orquestra detect → refit → validate → promote/rollback)
- [ ] Tabela `calibration_health` em ORM + init.sql
- [ ] CLI `monitor-calibration [--dry-run]`
- [ ] K8s CronJob `calibration-monitor` (3h UTC daily)
- [ ] History de pickles em `/data/models/history/<timestamp>.pkl` (manter últimos 5)
- [ ] Testes unitários: `detect_drift` em 12 cenários (cooldown, insufficient, ok, warn, trigger, rollback)

### Fora do Escopo

- Alertas externos (Slack, email) — só log + tabela
- Dashboard frontend pra calibration health — só query SQL
- Retrain automático do CatBoost (v1.17)
- A/B testing entre calibrações — só validação com hold-out

## Research

Ver [[../research/calibration-monitoring|calibration-monitoring]]:

- ECE + accuracy combined gate é mais robusto que ECE isolado
- Thresholds absoluto (ECE > 0.05) + relativo (ECE > 2× baseline) cobrem drift gradual e súbito
- Cooldown de 7 dias evita feedback loops
- Hold-out validation antes de promover evita regressões
- Min n ≥ 40 evita ruído amostral (Brasileirão: ~4 rodadas)

Contexto histórico: v1.11.0 teve `ece_calibrated = 0.028` em 409 samples. Threshold `0.05` representa ~1.8× baseline — espaço pra flutuação normal sem alertar spurious.

## Testing

### Unit (domain puro — zero mocks)

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

- Rodar `monitor-calibration --dry-run` em dataset atual → verificar que não há drift (ECE baixo pós-v1.14)
- Injetar predições com confidences shiftadas → verificar que trigger dispara
- Simular re-fit com pickle pior → verificar rollback

## Success Criteria

- [ ] `monitor-calibration --dry-run` roda em < 30s em dataset completo
- [ ] 12+ testes unitários passando
- [ ] Domain layer zero infra deps (`calibration_monitor.py` só importa numpy + `domain.calibration`)
- [ ] CronJob K8s rodando, métricas em `calibration_health`
- [ ] Pickle antigo arquivado em `history/` antes de qualquer re-fit
- [ ] Rollback funcional: teste manual com pickle pior reverte automaticamente
- [ ] Zero regressão em `fit-calibration` manual — backward compat

## Próximos pitches ligados

- v1.17.0 — CatBoost hyperopt + SHAP pruning (tuning do modelo base, não da calibração)
- v1.18.0 (futuro) — Alertas externos (webhook) quando trigger ou rollback dispara
- v1.19.0 (futuro) — Dashboard frontend pra calibration health timeline
