---
tags:
  - pitch
  - statsbomb
  - pgvector
  - rapm
  - embeddings
  - xT
  - VAEP
  - pressing
  - viz
  - cli
  - k8s
---

# Pitch — State of the Art Analytics Engine (v0.2.0)

## Problema

O motor analítico atual opera em ~60% do padrão de mercado. Faltam as métricas e modelos que clubes de elite (Liverpool, Man City, Brighton) e empresas líderes (StatsBomb/Hudl, SciSports, Opta) consideram essenciais:

1. **Sem Possession Value Model** — Não temos xT nem VAEP. Toda ação no campo é avaliada apenas por contagem bruta (passes, chutes) ou xG direto. Não existe como medir o valor de um passe progressivo no meio-campo que não resultou em finalização. Clubes e empresas tratam um PSV model como requisito mínimo desde 2019.

2. **Pressing é uma contagem bruta** — Temos `pressures` e `pressure_regains`, mas não PPDA, pressing success rate, counter-pressing fraction, nem high turnovers. Liverpool define sua identidade tática pelo pressing (PPDA 9.89); nosso motor não consegue nem detectar isso.

3. **Similaridade ignora posição** — Embeddings comparam goleiro com atacante no mesmo espaço vetorial. Clustering tem apenas 6 arquétipos fixos sem validação estatística. O mercado usa embeddings separados por grupo posicional com 12-16 roles e silhouette analysis.

4. **RAPM simplificado** — Ridge simples com indicadores binários +1/-1. Sem prior bayesiano (SPM), sem split ofensivo/defensivo real, sem design weights, sem multi-season. O estado da arte (MBAPPE) resolve cada um desses problemas.

5. **Métricas incompletas** — Faltam progressive receptions, shot quality (PSxG), pass breakdown por tipo/distância, duel win rates com contexto, tackle success rate.

Research: [[market-standard-analytics]]

## Solução

Upgrade completo do motor analítico em 6 frentes, cada uma elevando o projeto ao padrão de mercado:

### A. Expected Threat (xT) — Possession Value Model

Implementação custom do modelo de Karun Singh:
- Grid **16×12** (192 zonas) sobre campo StatsBomb (120×80)
- Markov chain iterativo: `xT(x,y) = s(x,y)×g(x,y) + m(x,y)×Σ T(x,y→z,w)×xT(z,w)`
- Convergência em ~5 iterações
- Valor de cada ação = `xT(destino) - xT(origem)`
- Treinado sobre todas as partidas open data do StatsBomb para robustez
- **Novo módulo: `possession_value.py`** — separado do player_metrics para manter arquitetura limpa

### B. VAEP — ML-Based Action Valuation

Integração via lib `socceraction` (MIT license):
- Converter eventos StatsBomb → formato **SPADL** (Standardized Player Action Description Language)
- Treinar 2 modelos (gradient boosting): P(score) e P(concede) nos próximos 10 eventos
- **VAEP(ação) = ΔP(score) - ΔP(concede)**
- Features: tipo de ação, localização, bodypart, resultado + últimas 3 ações como contexto
- Complementa xT com consciência de contexto (contra-ataque > posse estéril)

### C. Pressing Metrics Suite

Extração completa de métricas de pressing a partir dos events StatsBomb:
- **PPDA** (Passes Per Defensive Action): passes adversários no terço defensivo/médio ÷ ações defensivas
- **Pressing success rate**: pressões que resultam em recuperação em ≤5s ÷ total
- **Counter-pressing fraction**: % de perdas de bola seguidas de pressão em ≤5s
- **High turnovers**: recuperações a ≤40m do gol adversário
- **Shot-ending high turnovers**: high turnovers que geram finalização em ≤15s
- **Pressing zones**: distribuição de pressão por 6 zonas horizontais do campo

### D. Position-Aware Embeddings & Archetypes

Reescrita do sistema de embeddings:
- **Grupos posicionais**: GK, DEF (CB, FB), MID (DM, CM, AM), FWD (W, ST)
- **Embeddings separados por grupo**: PCA e clustering rodam dentro de cada grupo
- **12-16 arquetipos contextualizados** por grupo (não mais 6 genéricos):
  - DEF: Playmaking CB, Stopper, Ball-Playing FB, Attacking FB
  - MID: Deep-Lying Playmaker, Box-to-Box, Defensive Mid, Creative AM
  - FWD: Target Man, Inside Forward, Complete Forward, Poacher
- **Silhouette analysis** para determinar K ótimo por grupo
- **Explained variance** reportada no PCA (% informação retida)
- **Posição primária** armazenada no banco (nova coluna `position_group`)
- Similaridade pgvector filtra por grupo posicional automaticamente

### E. RAPM Avançado (Bayesian + Split Off/Def)

Upgrade do modelo RAPM inspirado no MBAPPE:
- **Splints** ao invés de stints: quebrar também em gols marcados (mais observações)
- **SPM prior bayesiano**: usar métricas box-score já calculadas (goals, assists, tackles/90) como prior Ridge `β ~ N(β_SPM, τ²)`
- **Split ofensivo/defensivo**: duplicar variáveis de jogador — colunas separadas para impacto ofensivo e defensivo
- **Design weights**: pesos baseados em fração de toques/ações do jogador no stint (não binário +1/-1)
- **Multi-season**: combinar stints de múltiplas temporadas para estabilidade (quando disponível)
- **Cross-validation** para λ = σ²/τ² (balancear dados vs. prior)

### F. Métricas Detalhadas

Expansão do `player_metrics.py`:
- **Progressive receptions**: receber passe que avança ≥10 yards
- **Shot quality**: xG por chute, big chances (xG ≥ 0.3), big chances missed
- **Pass breakdown**: short (<15y), medium (15-30y), long (>30y) com success rate cada
- **Pass under pressure**: passes tentados/completados sob pressão adversária
- **Switches of play**: passes laterais >30 yards
- **Ground duel win rate**: duelos no chão ganhos / total
- **Tackle success rate**: tackles bem-sucedidos / total
- **Duel zones**: distribuição de duelos por terço do campo

## Arquitetura

### Módulos afetados

| Módulo | Ação | Descrição |
|--------|------|-----------|
| **`possession_value.py`** | **NOVO** | xT (custom) + VAEP (via socceraction). Treina modelos, valora ações, persiste no banco |
| **`pressing.py`** | **NOVO** | PPDA, pressing success, counter-pressing, high turnovers, zonas. Opera sobre events crus |
| `player_metrics.py` | MODIFICAR | Adicionar ~15 métricas (progressive receptions, shot quality, pass breakdown, duel detail) |
| `player_embeddings.py` | REESCREVER | Position-aware embeddings, grupos posicionais, arquetipos expandidos, silhouette analysis |
| `rapm.py` | REESCREVER | Splints, SPM prior, split off/def, design weights, multi-season |
| `db.py` | MODIFICAR | Novos modelos ORM (ActionValue, PressingMetrics, PlayerPosition), colunas novas em PlayerMatchMetrics e PlayerEmbedding |
| `viz.py` | MODIFICAR | Novos plots: xT heatmap, pressing zones, xT flow, shot map detalhado |
| `export.py` | MODIFICAR | Seções novas no scout report: possession value, pressing profile, position-aware percentiles |
| `cli.py` | MODIFICAR | Novos subcomandos e flags; integrar novos dados nos outputs existentes |

### Grafo de dependência (atualizado)

```
cli.py (orchestrator)
  ├── db.py (data layer — imports nothing from project)
  ├── player_metrics.py (extraction — statsbombpy only)
  ├── possession_value.py  ← NOVO (xT + VAEP — statsbombpy + socceraction + db)
  ├── pressing.py           ← NOVO (pressing metrics — statsbombpy + db)
  ├── network_analysis.py (graphs — statsbombpy + networkx + db)
  ├── player_embeddings.py (ML — sklearn + db)
  ├── rapm.py (stats — statsbombpy + sklearn + db)
  ├── viz.py (visualization — matplotlib + mplsoccer + networkx)
  └── export.py (reports — db + networkx)
```

### Schema

#### Novas tabelas

```sql
-- Valores de ação (xT e VAEP por evento)
CREATE TABLE IF NOT EXISTS action_values (
    match_id INTEGER REFERENCES matches(match_id),
    event_index INTEGER,
    player_id INTEGER,
    player_name VARCHAR(100),
    team VARCHAR(100),
    action_type VARCHAR(50),       -- Pass, Carry, Shot, Dribble, etc.
    start_x REAL, start_y REAL,
    end_x REAL, end_y REAL,
    xt_value REAL,                 -- xT delta (end - start)
    vaep_value REAL,               -- VAEP delta (ΔP_score - ΔP_concede)
    vaep_offensive REAL,           -- ΔP_score only
    vaep_defensive REAL,           -- ΔP_concede only
    PRIMARY KEY (match_id, event_index)
);

-- Métricas de pressing por time por partida
CREATE TABLE IF NOT EXISTS pressing_metrics (
    match_id INTEGER REFERENCES matches(match_id),
    team VARCHAR(100),
    ppda REAL,
    pressing_success_rate REAL,
    counter_pressing_fraction REAL,
    high_turnovers INTEGER,
    shot_ending_high_turnovers INTEGER,
    pressing_zone_1 REAL,          -- % de pressões por zona (6 zonas)
    pressing_zone_2 REAL,
    pressing_zone_3 REAL,
    pressing_zone_4 REAL,
    pressing_zone_5 REAL,
    pressing_zone_6 REAL,
    PRIMARY KEY (match_id, team)
);

-- Posição primária de cada jogador por temporada (para embeddings)
-- Adicionada como coluna em player_embeddings
```

#### Colunas novas em tabelas existentes

```sql
-- player_match_metrics: ~15 novas colunas
ALTER TABLE player_match_metrics ADD COLUMN IF NOT EXISTS
    progressive_receptions REAL,
    big_chances REAL,             -- shots com xG >= 0.3
    big_chances_missed REAL,
    passes_short REAL,            -- < 15y
    passes_short_completed REAL,
    passes_medium REAL,           -- 15-30y
    passes_medium_completed REAL,
    passes_long REAL,             -- > 30y
    passes_long_completed REAL,
    passes_under_pressure REAL,
    passes_under_pressure_completed REAL,
    switches_of_play REAL,
    ground_duels_won REAL,
    ground_duels_total REAL,
    tackle_success_rate REAL,
    xt_generated REAL,            -- soma xT das ações do jogador
    vaep_generated REAL,          -- soma VAEP das ações do jogador
    pressing_success_rate REAL;   -- individual

-- player_embeddings: posição e dimensão expandida
ALTER TABLE player_embeddings ADD COLUMN IF NOT EXISTS
    position_group VARCHAR(10);   -- GK, DEF, MID, FWD
-- Embedding dimension: 16 → 16 (mantém, mas separado por grupo)

-- stints: renomear conceito para splints (quebra em gols também)
-- Manter tabela e adicionar coluna de tipo
ALTER TABLE stints ADD COLUMN IF NOT EXISTS
    boundary_type VARCHAR(20);    -- substitution, goal, period_start
```

#### Novos índices

```sql
CREATE INDEX IF NOT EXISTS idx_action_values_match ON action_values(match_id);
CREATE INDEX IF NOT EXISTS idx_action_values_player ON action_values(player_id);
CREATE INDEX IF NOT EXISTS idx_pressing_metrics_match ON pressing_metrics(match_id);
```

### Infra (K8s)

- Atualizar `k8s/configmap.yaml` com novas tabelas/colunas no `init.sql`
- Sem mudança em deployment, PVC ou service
- Considerar aumentar PVC se volume de dados crescer muito com action_values (1 row por evento)

### Novas dependências

```toml
# pyproject.toml
[project.dependencies]
# ... existentes ...
socceraction = ">=1.5"     # SPADL + xT + VAEP
xgboost = ">=2.0"          # VAEP model backend (gradient boosting)
```

## Escopo

### Dentro do Escopo

- [ ] **Módulo `possession_value.py`**: xT custom (grid 16×12, Markov chain) + VAEP (via socceraction, gradient boosting)
- [ ] **Módulo `pressing.py`**: PPDA, pressing success rate, counter-pressing fraction, high turnovers, shot-ending HT, zonas
- [ ] **Reescrita `player_embeddings.py`**: grupos posicionais, embeddings separados, 12-16 archetipos, silhouette analysis, explained variance
- [ ] **Reescrita `rapm.py`**: splints, SPM prior bayesiano, split off/def, design weights
- [ ] **Expansão `player_metrics.py`**: +15 métricas (progressive receptions, shot quality, pass breakdown, duel detail, switches)
- [ ] **Schema update `db.py`**: tabelas action_values + pressing_metrics, colunas novas em player_match_metrics e player_embeddings
- [ ] **Schema update `k8s/configmap.yaml`**: init.sql sincronizado com db.py
- [ ] **Viz novas `viz.py`**: xT heatmap, pressing zones, shot map detalhado
- [ ] **Export update `export.py`**: seções de possession value, pressing profile, percentiles por posição
- [ ] **CLI integration `cli.py`**: novos dados integrados nos outputs existentes
- [ ] **Testes unitários** para cada módulo novo/modificado
- [ ] **Testes de integração** com PostgreSQL para novas tabelas e queries

### Fora do Escopo

- EPV (Expected Possession Value) — requer tracking data que não temos
- OBV (On-Ball Value) — modelo proprietário StatsBomb/Hudl
- Football2Vec / GCN embeddings — complexidade alta, impacto incremental
- Cross-league normalization — requer dados multi-liga processados
- Frontend/dashboard — será pitch separado
- Video annotation linking
- Market value / contract integration

## Research Necessária

- [x] Padrão de mercado em football analytics — [[market-standard-analytics]]
- [ ] Validação do xT: comparar nossa implementação com valores de referência publicados (Euro 2024, Premier League)
- [ ] VAEP: testar socceraction com StatsBomb open data, verificar compatibilidade de versões
- [ ] Pressing: mapear todos os campos de pressure events do StatsBomb para confirmar viabilidade de cada métrica
- [ ] Posições: definir mapeamento StatsBomb position → grupo posicional (GK/DEF/MID/FWD)
- [ ] RAPM priors: testar estabilidade do SPM prior com dados open data (amostra menor que ligas completas)

## Estratégia de Testes

### Unitários (pytest + pytest-mock)

- **possession_value.py**: testar convergência do xT com dados sintéticos (grid 4×3), valores conhecidos; testar VAEP pipeline com eventos mock
- **pressing.py**: testar PPDA com DataFrames de passes/defensivas com contagens conhecidas; testar counter-pressing detection com timestamps mock
- **player_metrics.py**: testar cada métrica nova isoladamente com eventos fabricados; edge cases (0 minutos, sem duelos, etc.)
- **player_embeddings.py**: testar clustering por grupo posicional; silhouette score com dados sintéticos; verificar que GK não aparece em FWD cluster
- **rapm.py**: testar splint boundary em gols; testar SPM prior vs. sem prior; testar design weights vs. binário

### Integração (pytest + PostgreSQL de teste)

- Novas tabelas: INSERT/SELECT em action_values e pressing_metrics
- pgvector: similaridade filtrada por position_group
- Pipeline completo: extract → possession_value → persist → query
- Schema sync: verificar que db.py ORM e init.sql produzem schemas idênticos

### Manual

- Rodar `moneyball analyze-match <match_id>` e verificar xT/VAEP/pressing nos outputs
- Comparar xT values de uma partida conhecida (ex: final Champions League) com valores publicados
- Verificar que `find-similar` agora agrupa por posição
- Comparar RAPM rankings antes/depois do upgrade

## Critérios de Sucesso

- [ ] xT heatmap converge e produz superfície de ameaça consistente com literatura (zona central perto da área = maior xT)
- [ ] VAEP treina com sucesso sobre dados open data e valora ações defensivas (não apenas ofensivas)
- [ ] PPDA de times conhecidos (ex: Barcelona La Liga 2018/19) está dentro de ±1.0 de valores publicados
- [ ] Similaridade de jogadores retorna apenas jogadores da mesma posição (ou opt-in para cross-position)
- [ ] RAPM com SPM prior produz ranking mais estável que RAPM simples (menor variância entre temporadas)
- [ ] Todas as novas métricas persistidas no PostgreSQL e consultáveis via CLI
- [ ] Zero regressão nos comandos existentes (analyze-match, compare-players, etc.)
- [ ] Cobertura de testes unitários ≥ 80% nos módulos novos
- [ ] `python3 -m py_compile football_moneyball/*.py` passa sem erros
