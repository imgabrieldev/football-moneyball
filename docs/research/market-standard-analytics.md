---
tags:
  - research
  - analytics
  - xT
  - VAEP
  - OBV
  - pressing
  - RAPM
  - embeddings
  - market-standard
---

# Research — Padrão de Mercado em Football Analytics

> Research date: 2026-04-03
> Sources: [listadas ao final]

## Context

O motor analítico do Football Moneyball está em ~60% do padrão de mercado. Esta pesquisa mapeia o que clubes de elite (Liverpool, Man City, Brighton, Brentford) e empresas de analytics (StatsBomb/Hudl, Opta/Stats Perform, SciSports) efetivamente usam, para priorizar os gaps mais impactantes.

---

## 1. Possession Value Models — O Core do Analytics Moderno

### O que o mercado usa

Toda empresa séria de analytics tem um **Possession State Value (PSV) model** — a métrica que responde "quanto vale cada ação no campo?". Existem 5 variantes principais:

| Modelo | Criador | Abordagem | Dados necessários |
|--------|---------|-----------|-------------------|
| **xT** (Expected Threat) | Karun Singh, 2018 | Grid 16×12, Markov chain, iterativo | Event stream |
| **VAEP** | KU Leuven (Tom Decroos) | ML (gradient boosting), últimas 3 ações | Event stream (SPADL) |
| **OBV** (On-Ball Value) | StatsBomb/Hudl | 2 modelos separados (GF/GA), xG-trained | Event stream + pressure |
| **EPV** (Expected Possession Value) | Javier Fernández | State-of-the-art, off-ball | **Tracking data** |
| **g+** (Goals Added) | American Soccer Analysis | ML, endpoint-based | Event stream |

### xT — Expected Threat (Karun Singh)

O modelo mais acessível e amplamente implementado. Funciona assim:

1. **Grid 16×12** divide o campo em 192 zonas
2. Para cada zona calcula:
   - `s(x,y)` — probabilidade de chutar daquela zona
   - `m(x,y)` — probabilidade de passar/conduzir
   - `g(x,y)` — probabilidade de gol ao chutar dali
   - `T(x,y→z,w)` — matriz de transição (probabilidade de mover para cada outra zona)
3. **Equação iterativa:**
   ```
   xT(x,y) = [s(x,y) × g(x,y)] + [m(x,y) × Σ T(x,y→z,w) × xT(z,w)]
   ```
4. Converge em ~5 iterações
5. **Valor da ação** = `xT(destino) - xT(origem)`

**Implementável com StatsBomb open data.** Não precisa de tracking. Já existe a lib `socceraction` (Python, MIT license) que implementa xT e VAEP com loaders pra StatsBomb.

### VAEP — Valuing Actions by Estimating Probabilities

Mais sofisticado que xT:

- Usa as **últimas 3 ações** como contexto (não apenas localização)
- Treina 2 modelos ML: P(score) e P(concede) nos próximos N eventos
- **VAEP(ação) = ΔP(score) - ΔP(concede)**
- Features: tipo de ação, localização, bodypart, resultado, contexto
- Formato **SPADL** padroniza eventos de qualquer provider

**Vantagem sobre xT:** captura contexto (contra-ataque vale mais que posse estéril). **Desvantagem:** precisa treinar modelo ML, mais complexo.

### OBV — On-Ball Value (StatsBomb/Hudl)

O padrão comercial do StatsBomb:

- **2 modelos separados**: Goals For e Goals Against (não apenas net)
- Treinado em **xG do StatsBomb** (não gols reais — reduz variância)
- Features: localização (x, y, distância/ângulo ao gol), contexto (set piece vs open play), **pressão defensiva**, bodypart
- **Exclui** histórico de posse (evita bias por estilo de jogo/força do time)
- **Inclui ações defensivas e de goleiro**
- Receptores de passe **não recebem crédito** direto

**Key insight:** OBV separa valor ofensivo de defensivo, permitindo avaliar jogadores em ambas as dimensões independentemente.

### Qual implementar?

Para o Football Moneyball, a recomendação é:

1. **xT primeiro** — implementação simples, alto impacto, dados já disponíveis
2. **VAEP segundo** — via `socceraction` lib, ML-based, mais preciso
3. OBV/EPV são proprietários ou precisam de tracking data

---

## 2. Pressing Metrics — O Gap Mais Visível

### O que o mercado mede

| Métrica | Definição | Quem usa |
|---------|-----------|----------|
| **PPDA** | Passes permitidos por ação defensiva (menor = mais intenso) | Todos |
| **Counter-pressing fraction** | % de posses onde pressão é aplicada em ≤5s após perda | StatsBomb |
| **Pressing success rate** | % de pressões que resultam em recuperação | StatsBomb, clubes |
| **High turnovers** | Recuperações a ≤40m do gol adversário | Opta, WhoScored |
| **Shot-ending high turnovers** | High turnovers que geram finalização | StatsBomb |
| **Pressing intensity** | Nº de jogadores envolvidos no counter-press | Tracking-based |

### Dados do StatsBomb disponíveis

O StatsBomb define "pressure event" como jogador **a ≤5 yards** de adversário com bola (expande até 10y para goleiros). Os dados incluem:
- Jogadores envolvidos
- Localização no campo
- Duração da pressão
- Resultado (recuperação, falta, saída)

### Referências de clubes

- **Liverpool** (Slot): PPDA médio de 9.89 — o mais baixo da Premier League 2024/25
- **Man City** (Guardiola): PPDA ~8.3 na temporada 2017/18, "regra dos 6 segundos"
- Correlação pressing vs. posse: **r = 0.86**

### O que falta no nosso motor

Temos apenas `pressure_events` (contagem bruta) e `pressure_regains`. Faltam:
- PPDA (calculável: passes adversários / ações defensivas)
- Pressing success rate (pressões → recuperação / total pressões)
- Counter-pressing fraction (pressão em ≤5s após perda)
- High turnovers (recuperações no terço final)
- Zonificação do pressing (6 zonas horizontais)

---

## 3. RAPM Avançado — De Ridge Simples a MBAPPE

### Nosso RAPM atual

Ridge regression simples: `stint × player` matrix, target = xG differential, CV alpha.

### MBAPPE — O estado da arte

**M**ulti-League, **B**ayesian, **A**djusted and **P**enalized **P**lus-Minus **E**stimate:

1. **Multi-season** (2017-2022): mais dados → menos colinearidade
2. **Splints** (não stints): segmentos entre substituições **ou gols** — mais observações
3. **xG modificado**: ajusta valor do chute pela habilidade do finalizador
4. **SPM como prior bayesiano**: usa box-score stats (gols, assists, tackles/90) como prior para regularizar o RAPM
5. **Ridge com prior**: `λ = σ²/τ²` balanceia dados vs. prior informativo
6. **Split ofensivo/defensivo**: duplica variáveis para estimar impacto separado
7. **Design-weighted**: pesos baseados em localização de toques/passes (não binário 1/-1)
8. **Ajuste multi-liga**: normaliza métricas entre ligas para comparação cross-league

### CMU Soccer RAPM (acadêmico)

- ~4.000 stints por temporada (380 jogos Premier League)
- FIFA ratings como prior externo
- xG difference/90 como target
- Stint duration como peso

### O que melhorar no nosso

1. **Splints** ao invés de stints (quebrar também em gols)
2. **SPM prior** usando métricas box-score que já computamos
3. **Split ofensivo/defensivo** separado
4. **Multi-season** para estabilidade
5. **Design weights** por localização de ações (não binário)

---

## 4. Position-Aware Embeddings — Comparação Justa

### Problema atual

Nosso embedding compara todos os jogadores no mesmo espaço vetorial. Um goleiro pode ser "similar" a um atacante.

### Abordagens do mercado

#### Clustering por posição (Padrão)
- **4 clusters por grupo posicional** (defensores, meias, atacantes)
- Defensores: "Playmaking Defender", "Traditional CB", "Balanced Defender", "Attacking FB"
- Meias: "Deep-lying Playmaker", "Box-to-box", "Creative AM", "Defensive Mid"
- Atacantes: "Target Man", "Inside Forward", "Complete Forward", "Poacher"

#### Football2Vec (Ofir Magdaci)
- **Doc2Vec**: ações tokenizadas como "palavras", partida como "documento"
- **PlayerMatch2Vec**: vetor 32-dim por jogador×partida
- **Player2Vec**: média dos vetores de todas as partidas
- Posição capturada implicitamente nos padrões de ação
- Visualização via UMAP mostra clusters naturais por posição

#### Graph Convolutional Networks
- Player-similarity graph com cosine distance
- GCN gera embeddings que capturam relações topológicas
- Usado para recomendação de transferências

### O que implementar

1. **Filtro por posição** antes de calcular similaridade (mínimo viável)
2. **Embeddings separados por grupo posicional** (mais robusto)
3. **Arquetipos expandidos**: de 6 para 12-16 roles contextualizados
4. **Explained variance** do PCA (reportar % de informação retida)
5. **Silhouette analysis** para determinar K ótimo no clustering

---

## 5. Métricas Complementares — Quick Wins

### Progressive Actions (já temos parcial)
- ✅ Progressive passes (>10 yards closer to goal)
- ✅ Progressive carries
- ❌ Progressive receptions (receber passe progressivo)
- ❌ Normalização contextual (progressivo no terço defensivo ≠ no final)

### Shot Quality
- ✅ xG básico
- ❌ Post-shot xG (PSxG) — qualidade do chute após disparo
- ❌ Shot placement analysis
- ❌ Big chances created/missed

### Dueling Detail
- ✅ Aerials won/lost
- ❌ Ground duel win rate
- ❌ Tackle success rate
- ❌ Duel context (terço do campo, sob pressão)

### Pass Breakdown
- ✅ Pass completion %
- ❌ Short/medium/long pass success
- ❌ Pass under pressure success
- ❌ Switches of play
- ❌ Through ball accuracy

---

## 6. Ferramentas Open Source do Ecossistema

| Ferramenta | O que faz | Usar? |
|------------|-----------|-------|
| **socceraction** | SPADL + xT + VAEP com StatsBomb loader | ✅ Sim — xT e VAEP prontos |
| **mplsoccer** | Viz de futebol (já usamos) | ✅ Já usamos |
| **statsbombpy** | API StatsBomb (já usamos) | ✅ Já usamos |
| **football2vec** | Embeddings Doc2Vec | ⚠️ Avaliar — abordagem alternativa |
| **kloppy** | Loader universal de event data | ⚠️ Avaliar — se quisermos multi-provider |

---

## Implications for Football Moneyball

### Priorização por impacto × esforço

| # | Feature | Impacto | Esforço | Prioridade |
|---|---------|---------|---------|------------|
| 1 | **xT model** (via socceraction ou custom) | 🔴 Alto | Médio | P0 |
| 2 | **Pressing metrics** (PPDA, success rate, high turnovers) | 🔴 Alto | Baixo | P0 |
| 3 | **Position-aware similarity** (filtro + embeddings separados) | 🔴 Alto | Médio | P0 |
| 4 | **RAPM com SPM prior** + split off/def | 🟡 Médio | Alto | P1 |
| 5 | **VAEP** (via socceraction) | 🟡 Médio | Médio | P1 |
| 6 | **Shot/pass/duel breakdowns** | 🟡 Médio | Baixo | P1 |
| 7 | **Multi-season RAPM** | 🟡 Médio | Médio | P2 |
| 8 | **Football2Vec embeddings** | 🟢 Baixo | Alto | P2 |

### Roadmap sugerido

**v0.2.0 — Engine Upgrade (P0)**
- xT model (custom, grid 16×12)
- Pressing metrics suite (PPDA, success rate, counter-press, high turnovers)
- Position-aware embeddings e arquetipos expandidos

**v0.3.0 — Advanced Models (P1)**
- VAEP integration
- RAPM com SPM prior bayesiano + split ofensivo/defensivo
- Métricas detalhadas (shot quality, pass breakdown, duel context)

**v0.4.0 — Research Grade (P2)**
- Multi-season RAPM
- Football2Vec / GCN embeddings
- Cross-league normalization

---

## Sources

- [Introducing Expected Threat (xT) — Karun Singh](https://karun.in/blog/expected-threat.html)
- [VAEP — Valuing Actions by Estimating Probabilities — KU Leuven](https://dtai.cs.kuleuven.be/sports/vaep/)
- [socceraction — SPADL + xT + VAEP library](https://github.com/ML-KULeuven/socceraction)
- [On-Ball Value (OBV) — Hudl/StatsBomb](https://www.hudl.com/blog/introducing-on-ball-value-obv)
- [OBV Explainer — Hudl](https://www.hudl.com/blog/statsbomb-on-ball-value)
- [StatsBomb Counter-Pressing Metrics](https://blogarchive.statsbomb.com/articles/soccer/how-statsbomb-data-helps-measure-counter-pressing/)
- [PPDA Explained — Coaches' Voice](https://learning.coachesvoice.com/cv/ppda-explained-passes-per-defensive-action/)
- [MBAPPE Ratings — Game Models](https://www.gamemodelsfootball.com/about/about-mbappe-ratings)
- [CMU Soccer RAPM Model](https://www.stat.cmu.edu/cmsac/sure/2022/showcase/soccer_rapm.html)
- [Possession Value Models — FiveThirtyEight](https://fivethirtyeight.com/features/possession-is-the-puzzle-of-soccer-analytics-these-models-are-trying-to-solve-it/)
- [Football2Vec — Ofir Magdaci](https://github.com/ofirmg/football2vec)
- [Dynamic Expected Threat (DxT) — MDPI](https://www.mdpi.com/2076-3417/15/8/4151)
- [Devin Pleuler Analytics Handbook](https://github.com/devinpleuler/analytics-handbook)
- [Edd Webster Football Analytics Collection](https://github.com/eddwebster/football_analytics)
- [Decoding Player Roles via Clustering](https://medium.com/@marwanehamdani/decoding-player-roles-a-data-driven-clustering-approach-in-football-764654afb45b)
- [xT vs VAEP Comparison — Tom Decroos](https://tomdecroos.github.io/reports/xt_vs_vaep.pdf)
