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

## Problema

Modelo v1.10.0 tem **Brier 0.2437** em 78 predições resolvidas (meta realista: 0.20, linha de bookmakers). Reliability diagram expôs overconfidence sistemática:

| Bin confiança | n | Predito | Real | Gap |
|---|---|---|---|---|
| 33-40% | 6 | 37.9% | 16.7% | +21pts |
| 40-50% | 24 | 44.9% | 50.0% | -5pts ✓ |
| 50-60% | 21 | 55.1% | 38.1% | +17pts |
| 60-70% | 9 | 66.0% | 55.6% | +10pts |
| **70%+** | **18** | **81.1%** | **38.9%** | **+42pts** 🔥 |

Platt scaling v1.9.0 (a=0.46, b=0.04) é tímido: comprime 81%→67% quando deveria cair pra ~40%.

## Solução

Adicionar 2 métodos alternativos de calibração + auto-seleção via CV Brier:

### 1. Isotonic Regression (non-parametric)
- Qualquer distorção monotônica, não assume sigmoid como Platt
- Precisa n≥1000 pra não overfittar
- Tende a dominar quando há dados suficientes

### 2. Temperature Scaling (1-param, safe)
- `p_cal = p^(1/T) / Z` — single parameter
- Risco mínimo de overfitting
- Fallback seguro pra n pequeno

### 3. Auto-selection
- Time-split 80/20 (últimos 20% são val)
- Fit 3 métodos no train, avaliar Brier + ECE no val
- Escolher melhor, re-fittar no dataset completo

## Arquitetura

**Módulos afetados:**
- `domain/calibration.py` — 2 classes novas (`TemperatureScaler`, `IsotonicCalibrator`), 6 funções (fit + apply + métricas)
- `use_cases/fit_calibration.py` — refactor pra multi-method CV
- `use_cases/predict_all.py` — dispatch por método no `_apply_calibration`
- `cli.py` — flag `--method auto|platt|isotonic|temperature`

**Schema novo do pickle:**
```python
{
    "method": "platt" | "isotonic" | "temperature",
    "dixon_coles_rho": float,
    "platt_home/draw/away": {...},  # sempre salvo (fallback)
    "iso_home/draw/away": {...},    # se isotonic
    "temperature": {"T": float},    # se temperature
    "cv_results": {method: {brier_val, ece_val} for 3 methods},
    "metrics": {"brier_raw", "brier_calibrated", "ece_raw", "ece_calibrated", ...}
}
```

## Scope

**In:**
- Isotonic + Temperature + métricas ECE
- CV time-split + auto-select
- CLI flag
- Dispatch backward-compat (pickle antigo sem `method` default pra platt)
- 22 testes novos

**Out:**
- Ingestão de 2022/2023 (próximo step obvio — ataca root cause de n baixo)
- Beta calibration
- Hybrid temperature+isotonic residual

## Research

Baseado em `docs/research/calibration-methods.md`:
- Isotonic > Platt quando n ≥ 1000 (Sports AI blog, scikit-learn docs)
- Temperature scaling é post-hoc "millisecond fix" (Geoff Pleiss)
- **Calibração > Acurácia**: +34.69% ROI vs -35.17% (Wilkens, 2024)
- Bookmaker RPS benchmark: 0.17-0.22 (penaltyblog, 250M linhas)
- Meta <0.19 do pitch v1.6.0 é mais ambiciosa que bookmakers reais — ajustar pra 0.20

## Testing

Unit tests em `tests/test_domain_calibration.py`:
- TemperatureScaler: T=1 identity, T>1 compress, T<1 sharpen, normalização
- IsotonicCalibrator: identity, correção, monotonia, thresholds vazios
- fit_temperature: recupera T de dados sintéticos, reduz Brier em overconfident
- fit_isotonic_binary: monotonia, redução de Brier
- compute_ece: perfect = 0, overconfident > 0.4
- compute_brier_3class: formula checks

## Success Criteria

1. **Funcional**: `moneyball fit-calibration --method auto` mostra comparação CV e seleciona vencedor
2. **Tests**: 295+ passando (22 novos)
3. **Arch**: domain puro (só numpy/scipy/sklearn)
4. **Métricas reais (409 samples)**: Platt vence CV (Brier_val 0.652 vs Iso 0.695 vs Temp 0.673)
5. **In-sample improvements**: Brier -3%, **ECE -53%** (0.060→0.028), Acc +1.4pts

## Resultado

**SHIPPED 2026-04-05** — implementação completa, todos os testes passando.

**Descoberta empírica:** Com n=409 ainda insuficiente pra isotonic dominar (overfit no val split). Auto-selection funcionou corretamente escolhendo Platt. Próximo pitch natural: **ingestão 2022+2023** pra ultrapassar n=1000 e liberar isotonic.

**Comportamento em input extremo `[0.81, 0.10, 0.09]`:**
- Platt → 67.6%
- Isotonic → 83.0% (class-wise sem joint constraint, inflou)
- Temperature → 63.5%

## Próximos pitches ligados

- **v1.12.0 — Backfill histórico 2022/2023** (~760 matches adicionais) → re-fit calibração, isotonic deve dominar
- **v1.13.0 — Calibração monitorada** (cron de re-fit quando ECE > 0.015)
- **v1.14.0 — Beta calibration / hybrid** (se isotonic sozinho não atingir meta)
