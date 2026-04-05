---
tags:
  - research
  - calibration
  - probability
  - betting
  - brier
---

# Research — Probability Calibration em Modelos de Futebol

> Research date: 2026-04-05
> Sources: listadas ao final

## Context

Modelo atual do Football Moneyball apresenta Brier score 0.2437 em 78 predições resolvidas (meta pitch v1.6.0: <0.19). Análise de reliability diagram mostra overconfidence sistemática: no bin 70%+ o modelo prevê média 81.1% mas a taxa real de acerto é apenas 38.9% (gap de +42 pontos).

Platt scaling já está implementado (v1.9.0) mas os parâmetros atuais (a=0.46, b=0.04) comprimem de forma insuficiente — 81% vira apenas 67%. Pesquisa busca validar quais métodos de calibração são usados por bookmakers profissionais e clubes, e qual é a meta realista de Brier score.

## Findings

### 1. Benchmarks acadêmicos — o que é bom?

RPS (Ranked Probability Score) de bookmakers em 250M de linhas de odds:

| Liga | RPS típico |
|---|---|
| Eredivisie / Liga Portugal | 0.175–0.185 |
| Premier League | ~0.19 |
| Faixa geral europa | 0.175–0.224 |

**Bookmakers estão bem calibrados** — curvas H/D/A seguem a diagonal, sem viés sistemático. Pinnacle mantém overround de 2.39–3.00% (menor do mercado).

Meta realista: **Brier 0.20** (linha de bookmakers). Superar 0.19 exigiria bater Pinnacle consistentemente.

### 2. Calibração > Acurácia (para betting)

Paper ScienceDirect (Wilkens, 2024) comparou critérios de seleção de modelos:
- Modelo selecionado por **calibração**: +34.69% ROI
- Modelo selecionado por **acurácia**: −35.17% ROI

Conclusão: para betting, calibração é estritamente mais importante que % de acertos. Modelo que diz "55%" e acerta 55% das vezes é mais útil que modelo com maior accuracy mas probabilidades distorcidas.

### 3. Comparação dos métodos de calibração

| Método | Parâmetros | Ideal para | Risco overfit |
|---|---|---|---|
| **Platt scaling** | 2 (a, b) | n < 1000, distorção sigmoid | Baixo |
| **Isotonic regression** | non-parametric | n ≥ 1000, qualquer distorção monotônica | Médio |
| **Temperature scaling** | 1 (T) | NN logits, quick fix | Muito baixo |
| **Beta calibration** | 3 | tail skew assimétrico | Baixo-médio |

**Recomendação profissional (Sports AI)**: abordagem hybrid — temperature scaling primeiro, depois isotonic regression nos resíduos. Re-fit quando ECE (Expected Calibration Error) > 0.015.

Isotonic é superior a Platt quando há dados suficientes (~1000+ samples). Pode corrigir QUALQUER distorção monotônica, enquanto Platt assume distorção sigmoid.

### 4. xG model miscalibration — padrão conhecido

Análise de modelos Opta e StatsBomb (Tony ElHabr, 2024):
- **Bem calibrado** para xG < 0.25 (~90% dos chutes)
- **Sobrestima** para xG > 0.25

Este é exatamente o padrão do nosso modelo 1x2: calibrado em probabilidades moderadas, overconfident em probabilidades altas. Não é um bug único nosso — é viés característico de modelos stat-based de futebol.

StatsBomb XGBoost benchmark: AUC 0.878, Brier 0.069 em held-out shots.

### 5. Market blending — validado academicamente

Paper PLOS ONE (Betting Odds Rating System): consensus de bookmakers prevê MELHOR que Elo e FIFA ranking. Blend model+market é técnica padrão:

- Mercado fornece calibração forte (diagonal)
- Modelo captura ineficiências que o mercado missa
- Ensemble pondera ambas fontes

Nossa arquitetura v1.10.0 (market blending pós-calibração) está alinhada com state-of-the-art.

### 6. Dixon-Coles benchmark

Paper original 1997, >500 citações. EPL 2018-19 (272 matches): DC + player ratings melhora probabilidades em 53%+ dos casos vs. DC base. DC outperforma Poisson base em 15% de accuracy. Nosso pipeline já implementa (v1.9.0).

### 7. Profissionais (Brentford, Liverpool, Pinnacle)

- **Brentford**: Medallion architecture (Bronze→Silver→Gold), Airflow pipelines, foco em xG como predictor de recruitment, ML para injury risk
- **Liverpool**: departamento dedicado de PhDs, spatio-temporal models com tracking data, foco tático + recrutamento
- **Pinnacle**: frameworks especializados por esporte, erros logged + back-tested + re-treino contínuo, sharp bettors como input de pricing (feedback loop)

Profissionais usam calibração como pré-requisito básico — foco deles é modelagem avançada (tracking, GNN, Bayesian state-space).

## Implications for Football Moneyball

### Metas revisadas

| Métrica | Atual | Pitch v1.6.0 | Research-based |
|---|---|---|---|
| Brier 1x2 | 0.2437 | <0.19 | **0.20–0.21** (linha bookmaker) |
| Acc 1x2 | 42.3% | >54% | **50–54%** (limite prático) |

Meta <0.19 do pitch original é mais ambiciosa que bookmakers reais. Alvo realista: **Brier 0.20**.

### Ações priorizadas (research-backed)

**1. Trocar Platt → Isotonic regression (prioridade alta)**
- `sklearn.isotonic.IsotonicRegression`
- Validar com Brier CV (k=5)
- n=409 é limítrofe — fazer CV rigoroso
- Se overfit: cair pra temperature scaling

**2. Re-fit calibração com mais dados (prioridade alta)**
- Adicionar 2022 + 2023 ao training (+~760 matches)
- Com n>1000, isotonic fica seguro

**3. Hybrid temperature + isotonic (prioridade média)**
- Temperature scaling primeiro (1 param, safe)
- Isotonic nos resíduos
- Padrão recomendado por Sports AI blog

**4. Aceitar market blending como primary path (validado)**
- Não investir mais em fazer modelo base "perfeito" em isolamento
- Mercado é baseline forte; nosso valor é identificar edges

**5. Monitorar ECE além de Brier**
- Fit novo quando ECE > 0.015
- Implementar métrica no backtest

### O que NÃO fazer

- ❌ Shrinkage cego (puxar extremos pra média) — perde sinal
- ❌ Tentar superar <0.19 sem validar — pode ser overfitting do pitch target
- ❌ Investir em features contextuais novas antes de fixar calibração (v1.6.0 features já provaram inertes no feature importance)

## Sources

- [How Accurate Are Soccer Odds? — penaltyblog](https://pena.lt/y/2025/07/16/how-accurate-are-soccer-odds/)
- [Machine learning for sports betting: accuracy or calibration? — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S266682702400015X)
- [AI Model Calibration for Sports Betting — Sports AI](https://www.sports-ai.dev/blog/ai-model-calibration-brier-score)
- [Opta xG Model Calibration — Tony ElHabr](https://tonyelhabr.rbind.io/posts/opta-xg-model-calibration/)
- [scikit-learn Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Platt Scaling Complete Guide — Train in Data](https://www.blog.trainindata.com/complete-guide-to-platt-scaling/)
- [Temperature Scaling for NN Calibration — Geoff Pleiss](https://geoffpleiss.com/blog/nn_calibration.html)
- [Calibration Introduction Part 2 (Platt/Isotonic/Beta) — Abzu](https://www.abzu.ai/data-science/calibration-introduction-part-2/)
- [Dixon-Coles Model — Grokipedia](https://grokipedia.com/page/DixonColes_model)
- [Betting Odds Rating System — PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668)
- [Soccer Analytics Review 2025 — Jan Van Haaren](https://janvanhaaren.be/posts/soccer-analytics-review-2025/index.html)
- [How sharp are bookmakers? — DataGolf](https://datagolf.com/how-sharp-are-bookmakers)
- [The Brentford FC story — Sport Performance Analysis](https://www.sportperformanceanalysis.com/article/2018/6/8/the-history-of-brentford-football-analytics)
