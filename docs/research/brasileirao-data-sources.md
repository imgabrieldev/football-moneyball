---
tags:
  - research
  - brasileirao
  - data-sources
  - api
---

# Research — Fontes de Dados para o Brasileirão

> Research date: 2026-04-03
> Sources: [listadas ao final]

## Context

O StatsBomb open data não cobre o Brasileirão. Precisamos de uma fonte alternativa com dados da temporada atual e granularidade compatível com nosso schema (métricas por jogador por partida, xG, passes, duelos, pressão, posições).

## Findings

### Comparativo de APIs

| API | Brasileirão | Temporada atual | xG | Dados por jogador/partida | Eventos individuais | Preço/mês |
|-----|-------------|-----------------|-----|--------------------------|---------------------|-----------|
| **API-Football** | Sim | Sim | Parcial (time, não por chute) | Sim (shots, passes, tackles, dribbles, rating) | Não (agregado por jogador) | Free (100 req/dia) / $19+ |
| **Sportmonks** | Sim | Sim | Sim (add-on, por time e lineup) | Sim | Parcial | Trial 14 dias / pago |
| **Sofascore** (scraping) | Sim | Sim | Sim (shotmap com xG por chute) | Sim (heatmap, stats) | Sim (shotmap, heatmap coords) | Grátis (não oficial) |
| **FootyStats** | Sim | Sim | Sim (por time) | Não (agregado) | Não | $36/mês |
| **Understat** | **Não** | - | - | - | - | - |
| **FBref** (scraping) | Sim | Sim | Sim (por jogador/temporada) | Parcial | Não | Grátis (scraping) |

### 1. API-Football (api-sports.io) — Melhor custo-benefício

**Pontos fortes:**
- 1200+ competições incluindo Brasileirão Série A/B
- Endpoint `/fixtures/players` retorna por jogador por partida:
  - `shots.total`, `shots.on` (finalizações)
  - `goals.total`, `goals.assists`
  - `passes.total`, `passes.key`, `passes.accuracy`
  - `tackles.total`, `tackles.blocks`, `tackles.interceptions`
  - `duels.total`, `duels.won`
  - `dribbles.attempts`, `dribbles.success`
  - `fouls.drawn`, `fouls.committed`
  - `cards.yellow`, `cards.red`
  - `rating` (nota do jogo)
  - `minutes`, `position`
- Atualizado a cada minuto durante jogos ao vivo
- Lineups com posições

**Limitações:**
- **Sem xG por chute individual** — só agregado por time em endpoint separado
- **Sem eventos sequenciais** (não tem cada passe/carry individualmente)
- **Sem pressões/carries** como o StatsBomb tem
- Sem heatmap/coordenadas de ações

**Pricing:**
- Free: 100 requests/dia, temporada atual apenas
- Pro: $19/mês, 7.500 req/dia
- Ultra: $49/mês, 25.000 req/dia
- Mega: $99/mês, 75.000 req/dia

**Compatibilidade com nosso schema:** ~70%. Cobre a maioria das métricas de `player_match_metrics` mas falta xG por chute, pressões, carries, progressive passes/carries. Precisaria adaptar o pipeline.

### 2. Sofascore (via ScraperFC / API não oficial) — Mais dados, menos estável

**Pontos fortes:**
- **Shotmap com xG por chute** (coordenadas x,y + xG value)
- **Heatmap** com coordenadas de posicionamento
- Player stats detalhados por partida
- Cobre Brasileirão com dados completos
- Lib Python `ScraperFC` (pip install) ou `soccerdata`
- Grátis

**Limitações:**
- **API não oficial** — pode quebrar a qualquer momento
- Rate limiting agressivo
- Sem garantia de estabilidade
- Sem eventos sequenciais completos (passes individuais, carries)
- Scraping pode violar ToS

**Compatibilidade com nosso schema:** ~60%. Shotmap é excelente pra xG, mas falta granularidade de passes, pressões e carries individuais.

### 3. Sportmonks — Mais completo, mais caro

**Pontos fortes:**
- xG por time e por lineup (mais granular que API-Football)
- Brasileiro Série A coberto
- API estável e bem documentada
- Trial de 14 dias

**Limitações:**
- Preço não divulgado publicamente (enterprise)
- Sem eventos individuais tipo StatsBomb
- Necessita avaliação do trial pra confirmar granularidade

### 4. FBref (scraping via soccerdata) — Dados avançados gratuitos

**Pontos fortes:**
- Dados de xG, xA, progressive passes/carries por jogador
- Brasileirão coberto
- Lib Python `soccerdata` (pip install)
- **Dados powered by StatsBomb/Opta** — alta qualidade

**Limitações:**
- Dados **por temporada**, não por partida individual
- Scraping — sujeito a bloqueio
- Sem eventos individuais
- Atualização com delay

## Recomendação

### Melhor fit: **API-Football + Sofascore shotmap**

A combinação dá a melhor cobertura:

1. **API-Football** ($19/mês) como fonte principal:
   - Métricas por jogador por partida (shots, passes, tackles, dribbles, duels)
   - Lineups com posições
   - Atualização em tempo real
   - API estável e documentada

2. **Sofascore shotmap** (grátis, via ScraperFC) como complemento:
   - xG por chute individual (que API-Football não tem)
   - Coordenadas para heatmaps

### Mapeamento API-Football → nosso schema

| Nosso campo | API-Football field | Status |
|-------------|-------------------|--------|
| goals | goals.total | OK |
| assists | goals.assists | OK |
| shots | shots.total | OK |
| shots_on_target | shots.on | OK |
| xg | **Não disponível por jogador** | Precisa Sofascore |
| passes | passes.total | OK |
| passes_completed | passes.total * passes.accuracy/100 | Derivável |
| key_passes | passes.key | OK |
| tackles | tackles.total | OK |
| interceptions | tackles.interceptions | OK |
| blocks | tackles.blocks | OK |
| dribbles_attempted | dribbles.attempts | OK |
| dribbles_completed | dribbles.success | OK |
| fouls_committed | fouls.committed | OK |
| fouls_won | fouls.drawn | OK |
| minutes_played | games.minutes | OK |
| progressive_passes | **Não disponível** | Precisa StatsBomb/FBref |
| progressive_carries | **Não disponível** | Precisa StatsBomb/FBref |
| carries | **Não disponível** | Precisa StatsBomb/FBref |
| pressures | **Não disponível** | Precisa StatsBomb/FBref |
| touches | **Não disponível** | Parcial (FBref) |
| aerials_won/lost | duels.won / duels.total | Parcial |
| position | games.position | OK |

**Cobertura: ~18 de 30 métricas base mapeáveis diretamente.** As métricas avançadas (pressões, carries, progressive actions) não existem fora do StatsBomb/Opta.

## Implications for Football Moneyball

### O que precisaria mudar

1. **Novo módulo `data_providers/`** com interface abstrata — StatsBomb e API-Football como implementações
2. **Adapter pattern**: converter response da API-Football pro nosso schema `PlayerMatchMetrics`
3. **Métricas degradadas**: features como pressing, carries e progressive actions ficariam NULL para dados do Brasileirão
4. **xT não calculável**: sem eventos sequenciais com coordenadas, xT fica indisponível
5. **RAPM funciona**: stints podem ser reconstruídos via lineups + events da API-Football

### Custo estimado
- API-Football Pro: $19/mês (suficiente para ~25 partidas/dia)
- Para uma temporada completa do Brasileirão (~380 jogos): ~3.800 requests = 1 dia de cota

## Sources

- [API-Football](https://www.api-football.com/)
- [API-Football Documentation](https://api-sports.io/documentation/football/v3)
- [API-Football Pricing](https://www.api-football.com/pricing)
- [Sportmonks Brazilian Serie A API](https://www.sportmonks.com/football-api/serie-a-api-brazil/)
- [Sportmonks xG Data](https://www.sportmonks.com/football-api/xg-data/)
- [FootyStats Brazil Serie A xG](https://footystats.org/brazil/serie-a/xg)
- [FootyStats API](https://footystats.org/api)
- [ScraperFC — Python scraper](https://github.com/oseymour/ScraperFC)
- [soccerdata — Python scraper](https://github.com/probberechts/soccerdata)
- [Sofascore Corporate](https://corporate.sofascore.com/widgets)
- [FBref Serie A Stats](https://fbref.com/en/comps/24/Serie-A-Stats)
