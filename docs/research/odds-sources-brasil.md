---
tags:
  - research
  - betting
  - odds
  - scraping
  - betfair
  - betano
  - brasileirao
---

# Research — Fontes de Odds para o Brasileirão

> Research date: 2026-04-03
> Sources: [listadas ao final]

## Context

Precisamos de odds reais de casas de apostas brasileiras para substituir as odds sintéticas no backtesting e para o value-bets scanner funcionar com dados de mercado.

## Findings

### Ranking das opções (melhor → pior)

| # | Fonte | Tipo | Brasileirão | Grátis | Odds históricas | Manutenção |
|---|-------|------|-------------|--------|-----------------|------------|
| 1 | **Betfair Exchange API** | API oficial | Sim | Sim (conta) | Sim | Ativa |
| 2 | **OddsHarvester** (OddsPortal) | Scraper Python | Provável (100+ ligas) | Sim | Sim | Ativa (2025) |
| 3 | **The Odds API** | API comercial | Sim | 500 req/mês | Sim | Ativa |
| 4 | **Betano scraper** | Selenium | Sim (feito pra isso) | Sim | Não | Abandonado (2023) |
| 5 | **soccerapi** | Wrapper Python | Sim (Bet365, 888) | Sim | Não | Abandonado |

---

### 1. Betfair Exchange API — Melhor opção

**A única API oficial, gratuita e com dados de exchange (odds reais de mercado).**

- **Grátis** para uso pessoal (precisa de conta Betfair + app key)
- Lib Python: `betfairlightweight` (pip install)
- Soccer = event_type_id `1`
- Endpoints: `list_competitions`, `list_market_catalogue`, `list_market_book`
- **Odds de exchange** (não de bookmaker) — refletem probabilidade real do mercado
- **Brasileirão**: disponível se tiver mercado ativo na Betfair (principais jogos sim)
- **Dados ao vivo** + pré-jogo
- Autenticação: username + password + app_key + certificado SSL

**Como usar:**
```python
import betfairlightweight

trading = betfairlightweight.APIClient(
    username="...", password="...", 
    app_key="...", certs="/path/to/certs"
)
trading.login()

# Listar competições de futebol
comps = trading.betting.list_competitions(
    filter=betfairlightweight.filters.market_filter(event_type_ids=[1])
)
# Buscar odds
books = trading.betting.list_market_book(
    market_ids=["1.150038686"],
    price_projection=betfairlightweight.filters.price_projection(
        price_data=["EX_BEST_OFFERS"]
    )
)
```

**Vantagem:** Odds de exchange são mais eficientes que de bookmaker — se nosso modelo bater o exchange, temos edge real.

**Limitação:** Nem todo jogo do Brasileirão tem mercado na Betfair (jogos menores podem não ter liquidez).

---

### 2. OddsHarvester (OddsPortal scraper) — Melhor pra histórico

- Scraper de OddsPortal.com via Playwright
- **100+ ligas** (Brasileirão provável, precisa verificar slug)
- Coleta odds de **dezenas de casas** (Bet365, Betano, Pinnacle, 1xBet, etc.)
- **Odds históricas** por temporada inteira
- Output: JSON ou CSV
- CLI: `oddsharvester historic -s football -l brazil-serie-a --season 2025-2026 -m 1x2`
- Ativo em 2025, 100+ stars no GitHub

**Perfeito pra backtesting** — pega odds de abertura e fechamento de todas as casas por partida.

---

### 3. The Odds API — Já integrado (v0.4.0)

- 500 req/mês grátis, Brasileirão coberto
- h2h, totals, btts, spreads
- Odds históricas desde 2020
- Já temos adapter implementado

---

### 4. Betano Scraper — Específico pro Brasileirão

- Tutorial em PT-BR: `hansalemaos/tutorial_raspagem_de_dados_betano`
- Selenium + SeleniumBase
- Feito especificamente para Brasileirão Série A
- Coleta odds de apostas do site da Betano
- **Abandonado** (2023), pode não funcionar com site atual
- Abordagem frágil (Selenium quebrável com mudanças no site)

---

### 5. soccerapi — Wrapper simples

- `pip install soccerapi`
- Suporta Bet365, 888sport, Unibet
- Brasil listado como região suportada
- **Não mantido** — "some functionality may be broken"
- Simples de usar mas pouco confiável

---

## Recomendação

### Abordagem em 2 camadas:

**1. Betfair Exchange API** (tempo real + previsão)
- Adapter `adapters/betfair_provider.py`
- Odds de exchange = benchmark de mercado mais eficiente
- Grátis, API estável, lib Python madura
- Pra jogos com mercado ativo

**2. OddsHarvester** (backtesting + histórico)
- Script de coleta: baixar odds históricas do OddsPortal
- Dezenas de bookmakers por partida
- Comparar odds de abertura vs fechamento
- Calcular edge retroativo com odds REAIS

### O que isso melhora no modelo:

1. **Backtesting real** — trocar odds sintéticas por odds históricas reais de Bet365/Betano/Pinnacle
2. **Calibração** — comparar Brier score com odds de exchange (benchmark)
3. **Value bet scanning** — usar odds de exchange Betfair como "fair price" e comparar com bookmakers
4. **Arbitragem informacional** — se nosso xG diz algo diferente da Betfair, há oportunidade

## Sources

- [Betfair Developers](https://developer.betfair.com/)
- [Betfair Exchange API Guide](https://developer.betfair.com/exchange-api/)
- [betfairlightweight Python tutorial](https://betfair-datascientists.github.io/api/apiPythontutorial/)
- [Betfair API GitHub](https://github.com/betfair-datascientists/API)
- [OddsHarvester](https://github.com/jordantete/OddsHarvester)
- [soccerapi](https://github.com/S1M0N38/soccerapi)
- [Betano scraper tutorial](https://github.com/hansalemaos/tutorial_raspagem_de_dados_betano)
- [The Odds API](https://the-odds-api.com/)
- [The Odds API betting markets](https://the-odds-api.com/sports-odds-data/betting-markets.html)
