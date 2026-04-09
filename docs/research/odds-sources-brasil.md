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

# Research — Odds Sources for the Brasileirão

> Research date: 2026-04-03
> Sources: [listed at the end]

## Context

We need real odds from Brazilian bookmakers to replace the synthetic odds in backtesting and to make the value-bets scanner work with market data.

## Findings

### Ranking of options (best → worst)

| # | Source | Type | Brasileirão | Free | Historical odds | Maintenance |
|---|--------|------|-------------|------|-----------------|-------------|
| 1 | **Betfair Exchange API** | Official API | Yes | Yes (account) | Yes | Active |
| 2 | **OddsHarvester** (OddsPortal) | Python scraper | Likely (100+ leagues) | Yes | Yes | Active (2025) |
| 3 | **The Odds API** | Commercial API | Yes | 500 req/month | Yes | Active |
| 4 | **Betano scraper** | Selenium | Yes (built for it) | Yes | No | Abandoned (2023) |
| 5 | **soccerapi** | Python wrapper | Yes (Bet365, 888) | Yes | No | Abandoned |

---

### 1. Betfair Exchange API — Best option

**The only official, free API with exchange data (real market odds).**

- **Free** for personal use (requires a Betfair account + app key)
- Python library: `betfairlightweight` (pip install)
- Soccer = event_type_id `1`
- Endpoints: `list_competitions`, `list_market_catalogue`, `list_market_book`
- **Exchange odds** (not bookmaker odds) — reflect the real market probability
- **Brasileirão**: available if there is an active market on Betfair (main matches, yes)
- **Live data** + pre-match
- Authentication: username + password + app_key + SSL certificate

**How to use:**

```python
import betfairlightweight

trading = betfairlightweight.APIClient(
    username="...", password="...", 
    app_key="...", certs="/path/to/certs"
)
trading.login()

# List football competitions
comps = trading.betting.list_competitions(
    filter=betfairlightweight.filters.market_filter(event_type_ids=[1])
)
# Fetch odds
books = trading.betting.list_market_book(
    market_ids=["1.150038686"],
    price_projection=betfairlightweight.filters.price_projection(
        price_data=["EX_BEST_OFFERS"]
    )
)
```

**Advantage:** Exchange odds are more efficient than bookmaker odds — if our model beats the exchange, we have real edge.

**Limitation:** Not every Brasileirão match has a market on Betfair (smaller matches may lack liquidity).

---

### 2. OddsHarvester (OddsPortal scraper) — Best for historical data

- OddsPortal.com scraper via Playwright
- **100+ leagues** (Brasileirão likely, need to check the slug)
- Collects odds from **dozens of bookmakers** (Bet365, Betano, Pinnacle, 1xBet, etc.)
- **Historical odds** for entire seasons
- Output: JSON or CSV
- CLI: `oddsharvester historic -s football -l brazil-serie-a --season 2025-2026 -m 1x2`
- Active in 2025, 100+ stars on GitHub

**Perfect for backtesting** — gets opening and closing odds from all bookmakers per match.

---

### 3. The Odds API — Already integrated (v0.4.0)

- 500 req/month free, Brasileirão covered
- h2h, totals, btts, spreads
- Historical odds since 2020
- Adapter already implemented

---

### 4. Betano Scraper — Specific to the Brasileirão

- Tutorial in PT-BR: `hansalemaos/tutorial_raspagem_de_dados_betano`
- Selenium + SeleniumBase
- Built specifically for Brasileirão Série A
- Collects betting odds from the Betano site
- **Abandoned** (2023), may not work with the current site
- Fragile approach (Selenium breaks with site changes)

---

### 5. soccerapi — Simple wrapper

- `pip install soccerapi`
- Supports Bet365, 888sport, Unibet
- Brazil listed as a supported region
- **Not maintained** — "some functionality may be broken"
- Simple to use but unreliable

---

## Recommendation

### 2-layer approach:

**1. Betfair Exchange API** (real time + prediction)

- Adapter `adapters/betfair_provider.py`
- Exchange odds = most efficient market benchmark
- Free, stable API, mature Python library
- For matches with an active market

**2. OddsHarvester** (backtesting + history)

- Collection script: download historical odds from OddsPortal
- Dozens of bookmakers per match
- Compare opening vs closing odds
- Compute retroactive edge with REAL odds

### What this improves in the model:

1. **Real backtesting** — replace synthetic odds with real historical odds from Bet365/Betano/Pinnacle
2. **Calibration** — compare Brier score with exchange odds (benchmark)
3. **Value bet scanning** — use Betfair exchange odds as "fair price" and compare against bookmakers
4. **Informational arbitrage** — if our xG says something different from Betfair, there is an opportunity

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
