---
tags:
  - pitch
  - track-record
  - verify
  - automation
  - frontend
---

# Pitch — Track Record (v0.9.0)

## Problema

O sistema prevê jogos e identifica value bets, mas não rastreia se acertou ou errou. Hoje:

1. **Previsões são efêmeras** — `match_predictions` é sobrescrita a cada recompute. Não existe histórico de "o que previ na rodada 5".

2. **Verificação é manual** — precisa rodar `moneyball verify` e olhar no terminal. Não há comparação automática quando resultados chegam.

3. **Sem track record** — não sabemos: accuracy por rodada, evolução do Brier score ao longo do campeonato, quais mercados o modelo acerta mais (1X2 vs Over/Under), quais times o modelo erra sistematicamente.

4. **Value bets sem acompanhamento** — identificamos 59 value bets na rodada, mas depois que os jogos acontecem não sabemos quantas ganharam e qual foi o ROI real (não simulado).

5. **Sem confiança no modelo** — sem track record histórico, não dá pra saber se o modelo está melhorando ou piorando ao longo da temporada.

O sistema precisa ser **future-proof**: cada previsão é registrada, cada resultado é comparado, e o histórico completo fica acessível.

## Solução

### Lifecycle de uma previsão:

```
1. PREDICT  → Previsão salva com status "pendente" e rodada/data
2. MATCH    → Jogo acontece (resultado no Sofascore)
3. RESOLVE  → CronJob compara previsão vs resultado automaticamente
4. DISPLAY  → Frontend mostra histórico com acertos/erros
```

### Componentes:

#### A. Prediction History (tabela imutável)

Ao invés de sobrescrever `match_predictions`, cada previsão é um registro imutável:

```
prediction_history:
  id SERIAL
  match_id (hash dos times)
  home_team, away_team
  commence_time
  round (rodada)
  
  # Previsão do modelo
  home_win_prob, draw_prob, away_win_prob
  over_25_prob, btts_prob
  home_xg_expected, away_xg_expected
  most_likely_score
  predicted_at
  
  # Resultado real (preenchido depois)
  actual_home_goals (NULL até jogo acontecer)
  actual_away_goals
  actual_outcome (Home/Draw/Away)
  resolved_at
  status (pending → resolved)
  
  # Métricas de acerto
  correct_1x2 BOOLEAN
  correct_over_under BOOLEAN
  brier_score FLOAT
```

#### B. Value Bet History (tabela imutável)

Cada value bet identificada é registrada com resultado:

```
value_bet_history:
  id SERIAL
  prediction_id (FK → prediction_history)
  market, outcome
  model_prob, best_odds, bookmaker
  edge, kelly_stake
  
  # Resultado
  won BOOLEAN (NULL até resolver)
  profit FLOAT
  resolved_at
```

#### C. Auto-Resolve (use case)

Quando novos resultados são ingeridos do Sofascore, um use case `resolve_predictions` automaticamente:
1. Busca predictions com `status = 'pending'`
2. Para cada, verifica se o resultado existe no banco (`matches` table)
3. Se sim: preenche resultado, calcula brier, marca acerto/erro, muda status → `resolved`
4. Roda no CronJob de ingestão (após ingerir resultados)

#### D. Track Record API

```
GET /api/track-record              — resumo geral (accuracy, brier, ROI)
GET /api/track-record/predictions  — lista histórica de previsões
GET /api/track-record/value-bets   — lista histórica de value bets com P/L
GET /api/track-record/by-round     — accuracy por rodada
GET /api/track-record/by-team      — accuracy por time
GET /api/track-record/by-market    — accuracy por mercado (1X2, O/U, BTTS)
```

#### E. Frontend — Página Track Record

Nova página `/track-record` com:
- **Resumo**: accuracy geral, Brier, ROI real, total previsto/acertado
- **Evolução por rodada**: gráfico de linha (accuracy e Brier ao longo do tempo)
- **Por mercado**: qual tipo de aposta acertamos mais (tabela)
- **Por time**: quais times o modelo erra mais (tabela)
- **Histórico completo**: lista de cada previsão com resultado (scrollable)
- **Value bets P/L**: lista de cada aposta com ganho/perda

## Arquitetura

### Módulos afetados

| Módulo | Ação | Descrição |
|--------|------|-----------|
| `adapters/orm.py` | MODIFICAR | Novas tabelas PredictionHistory, ValueBetHistory |
| `adapters/postgres_repository.py` | MODIFICAR | CRUD pra histórico + queries de track record |
| `domain/track_record.py` | NOVO | Lógica de resolução (comparar pred vs resultado) |
| `use_cases/resolve_predictions.py` | NOVO | Auto-resolve quando resultados chegam |
| `use_cases/predict_all.py` | MODIFICAR | Salvar em prediction_history (imutável) |
| `use_cases/find_value_bets.py` | MODIFICAR | Salvar em value_bet_history |
| `api.py` | MODIFICAR | 6 novos endpoints de track record |
| `cli.py` | MODIFICAR | Comando `moneyball track-record` |
| `frontend/` | MODIFICAR | Nova página `/track-record` |
| `k8s/cronjob-ingest.yaml` | MODIFICAR | Rodar resolve após ingest |

### Schema

```sql
CREATE TABLE IF NOT EXISTS prediction_history (
    id SERIAL PRIMARY KEY,
    match_key INTEGER,
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    commence_time VARCHAR,
    round INTEGER,
    
    home_win_prob REAL,
    draw_prob REAL,
    away_win_prob REAL,
    over_25_prob REAL,
    btts_prob REAL,
    home_xg_expected REAL,
    away_xg_expected REAL,
    most_likely_score VARCHAR(10),
    predicted_at VARCHAR,
    
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    actual_outcome VARCHAR(10),
    resolved_at VARCHAR,
    status VARCHAR(20) DEFAULT 'pending',
    
    correct_1x2 BOOLEAN,
    correct_over_under BOOLEAN,
    brier_score REAL
);

CREATE TABLE IF NOT EXISTS value_bet_history (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES prediction_history(id),
    market VARCHAR(50),
    outcome VARCHAR(50),
    model_prob REAL,
    best_odds REAL,
    bookmaker VARCHAR(100),
    edge REAL,
    kelly_stake REAL,
    
    won BOOLEAN,
    profit REAL,
    resolved_at VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_pred_history_status ON prediction_history(status);
CREATE INDEX IF NOT EXISTS idx_pred_history_round ON prediction_history(round);
```

### Infra (K8s)

Modificar CronJob `ingest-sofascore` pra rodar `moneyball resolve` após ingestão:
```yaml
command: ["sh", "-c", "moneyball ingest --provider sofascore && moneyball resolve"]
```

## Escopo

### Dentro do Escopo

- [ ] Tabela `prediction_history` (imutável, 1 registro por previsão)
- [ ] Tabela `value_bet_history` (imutável, 1 registro por value bet)
- [ ] `domain/track_record.py` — resolve predictions vs resultados
- [ ] `use_cases/resolve_predictions.py` — auto-resolve
- [ ] `predict_all` salva em `prediction_history` ao invés de sobrescrever `match_predictions`
- [ ] `find_value_bets` salva em `value_bet_history`
- [ ] 6 endpoints API de track record
- [ ] CLI `moneyball resolve` e `moneyball track-record`
- [ ] Frontend página `/track-record` com resumo, gráficos, tabelas
- [ ] CronJob atualizado pra resolver após ingest
- [ ] Testes unitários pra resolução

### Fora do Escopo

- Alertas/notificações quando resultados chegam
- Comparação com outros modelos (benchmark)
- Export de relatório PDF
- Ajuste automático do modelo baseado em track record (meta-learning)

## Research Necessária

- [ ] Definir o que conta como "rodada" (Sofascore tem round info? ou inferir por data?)
- [ ] Definir matching entre prediction (nomes dos odds sem acento) e resultado (nomes Sofascore com acento) — já temos fuzzy match

## Estratégia de Testes

### Unitários
- `domain/track_record.py`: resolver prediction com resultado conhecido, brier calculado corretamente
- Prediction pending → resolved quando resultado chega
- Value bet won/lost calculado corretamente

### Integração
- Fluxo completo: predict → ingest resultado → resolve → check track record

### Manual
- Recomputar predictions, esperar rodada, ingerir, verificar track-record no frontend

## Critérios de Sucesso

- [ ] Predictions nunca são sobrescritas (histórico imutável)
- [ ] Após ingestão de resultados, predictions são resolvidas automaticamente
- [ ] `/track-record` mostra accuracy, Brier, ROI por rodada
- [ ] Value bets mostram P/L real
- [ ] Sem intervenção manual no fluxo predict → resolve
- [ ] Frontend acessível e informativo
