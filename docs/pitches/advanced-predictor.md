---
tags:
  - pitch
  - predictor
  - monte-carlo
  - xT
  - pressing
  - rapm
  - sofascore
  - brasileirao
---

# Pitch — Modelo Preditivo Avançado (v0.5.0)

## Problema

O modelo de previsão atual (`domain/match_predictor.py`) usa apenas **xG bruto** como input para o Monte Carlo, ignorando ~80% dos dados que já calculamos:

1. **Brier score de 0.76** — pior que aleatório (0.25). O modelo não está calibrado.
2. **Ignora xT** — temos Expected Threat calculado mas não alimenta o predictor. Um time que cria muito xT mas não finaliza bem é diferente de um que finaliza pouco mas com qualidade.
3. **Ignora pressing** — PPDA, counter-pressing e high turnovers já estão no banco mas não afetam a previsão. Times com pressing alto forçam mais erros → mais xG criado contra adversários.
4. **Ignora RAPM** — temos impacto individual de cada jogador mas não usamos pra ajustar xG pela escalação. Se o artilheiro está suspenso, o xG esperado deveria cair.
5. **Sem regressão à média** — Palmeiras com +9.1 gols acima do xG provavelmente vai regredir. O modelo deveria puxar overperformers em direção à média.
6. **Sem ingestão automática** — dados do Sofascore são ingeridos manualmente. O predictor deveria garantir dados frescos antes de prever.
7. **Bias pro Under** — o modelo prevê Under 2.5 em excesso porque xG médio do início do campeonato é baixo e não ajusta pela evolução da temporada.

### Dados que já temos no banco (87 partidas do Brasileirão 2026):

| Dado | Tabela | Usado no predictor? |
|------|--------|:---:|
| xG por jogador por partida | player_match_metrics | ✅ (parcial, só soma) |
| xA por jogador | player_match_metrics | ❌ |
| Gols reais vs xG | player_match_metrics | ❌ (regressão à média) |
| PPDA por time | pressing_metrics | ❌ |
| Pressing success rate | pressing_metrics | ❌ |
| Counter-pressing fraction | pressing_metrics | ❌ |
| High turnovers | pressing_metrics | ❌ |
| xT por ação (shotmap) | action_values | ❌ |
| Passes completados/tipo | player_match_metrics | ❌ |
| Big chances criadas | player_match_metrics | ❌ |
| Tackles/interceptions | player_match_metrics | ❌ |
| Posição do jogador | player_embeddings | ❌ |
| RAPM individual | (calculável) | ❌ |
| Embeddings/arquetipos | player_embeddings | ❌ |

Research: [[betting-value-model]], [[market-standard-analytics]]

## Solução

Reescrever `domain/match_predictor.py` com pipeline de 6 estágios que integra todos os dados disponíveis:

### Pipeline de Previsão

```
┌─────────────────────────────────────────────────┐
│ 1. INGESTÃO — Atualizar dados Sofascore         │
│    Buscar jogos novos desde última ingestão      │
│    Atualizar player_match_metrics + pressing     │
├─────────────────────────────────────────────────┤
│ 2. BASE xG — Histórico ponderado                │
│    xG ofensivo: média exponencial últimos 6 jogos│
│    xG defensivo: xGA médio do adversário        │
│    Fator casa calibrado (calcular do dataset)    │
├─────────────────────────────────────────────────┤
│ 3. AJUSTES TÁTICOS                              │
│    + Pressing: PPDA baixo → mais xG criado       │
│    + xT: times com alto xT criam mais perigo     │
│    + Big chances: taxa de conversão de grandes    │
│      chances ajusta over/under                   │
├─────────────────────────────────────────────────┤
│ 4. AJUSTE DE ESCALAÇÃO                          │
│    + RAPM dos titulares confirmados              │
│    + Jogador-chave ausente → penalidade no xG    │
│    + Top scorer do time suspenso → impacto       │
├─────────────────────────────────────────────────┤
│ 5. REGRESSÃO À MÉDIA                            │
│    Gols reais vs xG na temporada                 │
│    Se time marca muito acima do xG → reduzir     │
│    Se marca abaixo → aumentar                    │
│    Fator: puxar 30-50% em direção à média        │
├─────────────────────────────────────────────────┤
│ 6. MONTE CARLO — Simulação final                │
│    λ_home = xG ajustado (etapas 2-5)            │
│    λ_away = xG ajustado (etapas 2-5)            │
│    Poisson(λ) × 10.000 simulações               │
│    → P(1X2), P(O/U), P(BTTS), placares          │
└─────────────────────────────────────────────────┘
```

### Detalhamento dos ajustes

#### Estágio 2: Base xG

```python
def compute_base_xg(team_history, opponent_defense, is_home):
    # Média exponencial (decay=0.85, últimos 6 jogos)
    base_xg = weighted_avg(team_history.xg, decay=0.85)
    
    # Força defensiva do adversário
    # Se adversário sofre mais xG que a média → boost nosso xG
    league_avg_xga = mean(all_teams.xga)
    defense_factor = opponent.xga / league_avg_xga
    base_xg *= defense_factor
    
    # Fator casa calibrado do Brasileirão
    # Calcular empiricamente: mean(home_xg) - mean(away_xg)
    home_boost = calibrated_home_advantage  # ~0.25-0.35
    if is_home:
        base_xg += home_boost
    
    return base_xg
```

#### Estágio 3: Ajustes Táticos

```python
def apply_tactical_adjustments(base_xg, team_pressing, opponent_pressing, team_xt):
    adjusted = base_xg
    
    # Pressing: times com PPDA baixo (pressão alta) forçam mais turnovers
    # Correlação: PPDA adversário baixo → nosso xG diminui
    league_avg_ppda = mean(all_teams.ppda)
    pressing_factor = opponent.ppda / league_avg_ppda
    # PPDA alto do adversário (pressão fraca) → boost nosso xG
    adjusted *= (0.85 + 0.15 * pressing_factor)
    
    # xT: qualidade de criação de jogadas
    # Times com alto xT por partida criam mais perigo
    league_avg_xt = mean(all_teams.xt_per_match)
    xt_factor = team.xt_per_match / league_avg_xt
    adjusted *= (0.90 + 0.10 * xt_factor)
    
    # Counter-pressing: teams que recuperam rápido criam mais
    if team.counter_pressing_fraction > 50:
        adjusted *= 1.03  # +3% boost
    
    return adjusted
```

#### Estágio 4: Ajuste de Escalação

```python
def apply_lineup_adjustment(base_xg, confirmed_lineup, team_rapm):
    # Se temos lineup confirmada (Sofascore ~1h antes)
    if confirmed_lineup:
        # Somar RAPM dos 11 titulares
        lineup_rapm = sum(rapm[player_id] for player_id in confirmed_lineup)
        # Comparar com RAPM médio do time na temporada
        avg_rapm = mean(all_lineups_rapm)
        rapm_delta = lineup_rapm - avg_rapm
        # Ajustar xG: cada 0.1 de RAPM ≈ 0.05 xG
        base_xg += rapm_delta * 0.5
    
    return max(base_xg, 0.1)
```

#### Estágio 5: Regressão à Média

```python
def apply_regression_to_mean(xg_estimate, team_season_stats):
    # Gols reais vs xG na temporada
    goals = team_season_stats.total_goals
    xg_total = team_season_stats.total_xg
    overperformance = (goals - xg_total) / matches_played  # por jogo
    
    # Puxar 40% em direção à média (research: ~50% de overperformance é sorte)
    regression_factor = 0.40
    adjustment = -overperformance * regression_factor
    
    return max(xg_estimate + adjustment, 0.1)
```

### Auto-ingestão pré-previsão

Antes de cada previsão, o use case `PredictMatch` verifica se há jogos novos no Sofascore que ainda não foram ingeridos:

```python
class PredictMatch:
    def execute(self, match_id, home, away):
        # Passo 0: garantir dados frescos
        self._auto_ingest_if_needed()
        
        # Passo 1-6: pipeline de previsão
        ...
    
    def _auto_ingest_if_needed(self):
        last_match = repo.get_latest_match_date()
        if (today - last_match).days >= 1:
            sofascore = SofascoreProvider()
            new_matches = sofascore.get_matches(...)
            # ingerir novos jogos
```

## Arquitetura

### Módulos afetados

| Módulo | Ação | Descrição |
|--------|------|-----------|
| `domain/match_predictor.py` | **REESCREVER** | Pipeline 6 estágios com todos os ajustes |
| `domain/constants.py` | MODIFICAR | Adicionar constantes de calibração (home advantage, decay, regression factor) |
| `use_cases/predict_match.py` | MODIFICAR | Buscar pressing, xT, RAPM do repo + auto-ingestão |
| `use_cases/backtest.py` | MODIFICAR | Usar novo predictor, comparar Brier score antes/depois |
| `adapters/postgres_repository.py` | MODIFICAR | Novos queries: pressing por time, xT por time, RAPM season |
| `adapters/sofascore_provider.py` | MODIFICAR | Método para buscar apenas jogos novos (delta ingestion) |

### Schema

Sem mudanças no PostgreSQL. Todos os dados necessários já existem nas tabelas atuais:
- `player_match_metrics` — xG, xA, big_chances, passes, tackles
- `pressing_metrics` — PPDA, success rate, counter-pressing, high turnovers
- `action_values` — xT por ação (shotmap)
- `matches` — resultados reais para regressão à média

### Infra (K8s)

Sem mudanças.

## Escopo

### Dentro do Escopo

- [ ] Reescrever `domain/match_predictor.py` com pipeline de 6 estágios
- [ ] `compute_base_xg()` — média exponencial + força defensiva + fator casa calibrado
- [ ] `apply_tactical_adjustments()` — PPDA, xT, counter-pressing, big chances
- [ ] `apply_lineup_adjustment()` — RAPM dos titulares confirmados
- [ ] `apply_regression_to_mean()` — puxar overperformers em direção à média
- [ ] `calibrate_home_advantage()` — calcular empiricamente do dataset Brasileirão
- [ ] Atualizar `use_cases/predict_match.py` — buscar pressing, xT, RAPM + auto-ingestão
- [ ] Atualizar `use_cases/backtest.py` — usar novo pipeline, comparar Brier antes/depois
- [ ] Novos queries no repository — pressing médio por time, xT por time, RAPM season
- [ ] Delta ingestion no `sofascore_provider.py` — buscar apenas jogos novos
- [ ] Testes unitários para cada estágio do pipeline isoladamente
- [ ] Backtesting comparativo: modelo v0.4.0 vs v0.5.0 com mesmas 87+ partidas

### Fora do Escopo

- Machine learning (neural networks, gradient boosting) — manter Poisson + ajustes determinísticos
- Dados de tracking (posição em tempo real) — só event data do Sofascore
- Mercados de jogador (artilheiro, cartões) — só mercados de partida
- Novos data providers — usar Sofascore existente
- API REST — pitch separado (v0.6.0)

## Research Necessária

- [ ] Calibrar fator casa empírico do Brasileirão 2026 com nossas 87 partidas
- [ ] Medir correlação PPDA → xG criado no dataset (validar que pressing afeta xG)
- [ ] Medir correlação xT → gols (validar que xT é preditivo além do xG)
- [ ] Determinar regression factor ótimo (30%, 40%, 50%?) via backtesting
- [ ] Testar sensibilidade do modelo a cada ajuste isoladamente (ablation study)

## Estratégia de Testes

### Unitários (domain — zero mocks)
- `compute_base_xg()` — inputs conhecidos, output determinístico
- `apply_tactical_adjustments()` — PPDA alto vs baixo, xT alto vs baixo
- `apply_lineup_adjustment()` — com/sem lineup, RAPM positivo vs negativo
- `apply_regression_to_mean()` — overperformer vs underperformer
- `calibrate_home_advantage()` — dataset sintético com vantagem conhecida

### Integração
- Pipeline completo: 6 estágios encadeados produzem xG razoável (0.5-3.0)
- Backtesting v0.4.0 vs v0.5.0: Brier score deve diminuir

### Manual
- Comparar previsão de jogo específico com odds reais
- Verificar que auto-ingestão pega jogos novos do Sofascore
- Comparar Palmeiras (overperformer) antes/depois da regressão à média

## Critérios de Sucesso

- [ ] Brier score < 0.25 no backtesting (melhor que aleatório)
- [ ] Brier score do v0.5.0 < Brier score do v0.4.0 (melhoria mensurável)
- [ ] ROI do backtesting positivo com odds reais (The Odds API)
- [ ] Hit rate em value bets > 45%
- [ ] Cada estágio do pipeline tem impacto mensurável (ablation test)
- [ ] Auto-ingestão funciona sem intervenção manual
- [ ] Zero regressão nos comandos CLI existentes
- [ ] Todos os testes passam
