---
tags:
  - research
  - calibration
  - monitoring
  - drift
  - ece
  - mlops
---

# Research — Monitoramento e Retreino Automático de Calibração

> Research date: 2026-04-09
> Sources: listadas ao final

## Context

Em v1.11.0 ([[isotonic-calibration-v1.11]]) foi implementada auto-seleção entre Platt / Isotonic / Temperature scaling com pickle snapshot. O fit é manual — `moneyball fit-calibration` roda sob demanda. Em produção esse modelo degrada: distribuição de jogos muda (troca de técnicos, janela de transferências, lesões, fadiga de fim de temporada), e a calibração fica estale.

Objetivo: definir **quando** e **como** disparar um re-fit automático, e quais guardrails aplicar pra não re-calibrar em ruído.

## Findings

### 1. ECE como métrica de monitoramento

ECE (Expected Calibration Error) é a métrica canônica pra detectar drift de calibração quando labels são disponíveis com atraso — exatamente o caso do futebol, onde o resultado real sai ~2h depois da predição.

**Fórmula** (já implementada em [calibration.py:469](football_moneyball/domain/calibration.py#L469)):
```
ECE = Σ_b (|B_b|/n) · |acc_b − conf_b|
```

Bins de confiança (10 por padrão), weighted average do gap entre confidence média e accuracy empírica.

### 2. Thresholds na literatura

Não existe valor universal. A prática recomendada ([Medium: Angela Shi](https://medium.com/data-science-collective/a-practical-guide-to-prediction-drift-concept-drift-and-model-specific-monitoring-ee1ef086570b), [alldaystech](https://alldaystech.com/guides/artificial-intelligence/model-drift-detection-monitoring-response)) é:

1. **Estabelecer baseline**: ECE no momento do fit (training-set ECE após calibração). Em v1.11.0: ECE calibrated = 0.028.
2. **Definir threshold absoluto**: `ECE_limit = baseline + δ` onde `δ ≈ baseline` (doubling rule).
3. **Definir threshold relativo**: `ECE_live > 1.5× baseline` sustained.
4. **Minimum sample size**: não alertar com n < 30. Futebol: ~10 jogos/rodada → 3 rodadas mínimo.
5. **Sustained window**: drift precisa durar ≥ 2 medições consecutivas, não spike isolado.

**Proposta pro Moneyball:** ECE > 0.05 OU ECE > 2× baseline, sustained por 2 janelas de 20 predições resolvidas.

### 3. Accuracy-gated retrain

Drift de calibração SEM drift de accuracy é perigoso — pode ser só ruído amostral. A recomendação ([alldaystech](https://alldaystech.com/guides/artificial-intelligence/model-drift-detection-monitoring-response)) é combinar:

- ECE > threshold **E**
- Accuracy drop > 3pp vs baseline de 30 dias

Ambos precisam bater. Evita re-calibrar em cima de 20 jogos atípicos.

### 4. Cooldown e governance

Problemas comuns quando se automatiza retrain sem guardrails:

- **Feedback loops**: calibração se ajusta a ruído → pior ECE → re-ajusta. Cooldown mínimo de 7 dias entre re-fits.
- **Churn de parâmetros**: pickle muda toda semana, backtest fica incomparável. Sempre snapshot com timestamp.
- **Rollback**: se o novo fit piorar ECE out-of-sample na próxima janela, reverter pro pickle anterior.

### 5. Online isotonic

Isotonic regression pode ser re-fit incremental (scikit-learn `IsotonicRegression.fit` é O(n log n), fast o suficiente pra rodar full-refit em segundos pros ~500 samples acumulados). Não precisa de online learning sofisticado — re-fit from scratch dentro de um cronjob é suficiente.

### 6. Temperature scaling drift

Temperature scaling tem 1 parâmetro (T). Monitoramento direto: se `T_new / T_old > 1.3` ou `< 0.77`, probabilidades estão sendo comprimidas/esticadas significativamente → sinal de shift real na distribuição das confidences.

## Proposta de arquitetura

### Fluxo

```
CronJob a cada 24h
  │
  ├─ 1. Pegar predições resolvidas dos últimos 14 dias (ResolvedPrediction)
  │     com calibração aplicada no momento da predição
  │
  ├─ 2. Computar ECE_live, Brier_live, accuracy_live
  │
  ├─ 3. Comparar vs baseline (stored no calibration pickle metadata)
  │
  ├─ 4. Decisão:
  │     - ECE_live < threshold → OK, log métricas
  │     - ECE_live > threshold MAS accuracy estável → log warning, não age
  │     - ECE_live > threshold AND accuracy drop → TRIGGER re-fit
  │     - Cooldown ativo (< 7 dias desde último fit) → skip
  │
  ├─ 5. Se trigger:
  │     - Rodar fit_calibration() com dataset completo
  │     - Comparar novo ECE vs antigo em hold-out (últimos 20%)
  │     - Se melhor → salvar novo pickle, arquivar antigo em /data/models/history/
  │     - Se pior → alertar, manter antigo
  │
  └─ 6. Escrever métricas numa tabela calibration_health
```

### Guardrails

- Min n: 40 predições resolvidas desde o último re-fit
- Cooldown: 7 dias mínimo
- Hold-out validation: novo pickle precisa melhorar ECE em ≥ 10% OU manter within 5% do antigo
- History: manter últimos 5 pickles arquivados pra rollback
- Alertas: se 3 triggers consecutivos falharem (novo pior que antigo), pausar automação e logar incident

## Conclusões

1. **ECE + accuracy combined gate** é a regra mais robusta — evita re-calibrar em ruído
2. **Thresholds absolutos + relativos** pegam tanto drift gradual quanto súbito
3. **Cooldown** é mandatório pra evitar feedback loops
4. **Hold-out validation** do novo pickle é barato e evita regressões
5. **History + rollback** dá safety net

## Sources

- [A practical guide to prediction drift, concept drift, and model-specific monitoring — Angela Shi (Medium)](https://medium.com/data-science-collective/a-practical-guide-to-prediction-drift-concept-drift-and-model-specific-monitoring-ee1ef086570b)
- [Model Drift in Production: Detection, Monitoring & Response Runbook — AllDaysTech](https://alldaystech.com/guides/artificial-intelligence/model-drift-detection-monitoring-response)
- [Data Drift: Key Detection and Monitoring Techniques in 2026 — Label Your Data](https://labelyourdata.com/articles/machine-learning/data-drift)
- [Detect, Retrain, Repeat: Building a Model Drift Monitor with Alibi-Detect — Medium](https://medium.com/@myakalarajkumar1998/detect-retrain-repeat-building-a-model-drift-monitor-with-alibi-detect-ce1a32fe6cc5)
- [Classifier Calibration with ROC-Regularized Isotonic Regression — Berta et al., PMLR 2024](https://proceedings.mlr.press/v238/berta24a/berta24a.pdf)
- [[calibration-methods]] — pesquisa prévia em Platt/Isotonic/Temperature
