# Regime-Aware Genetic Algorithm for portfolio allocation

A genetic algorithm allocates a 26-asset, multi-currency portfolio (Indian stocks, US and global ETFs, commodities,
bonds and crypto), blended with Hierarchical Risk Parity, switched to HRP alone in volatility spikes, and scaled by
volatility. It is backtested walk-forward in rupees, net of costs, against HRP, equal weight, 60/40 and the NIFTY 50
ETF.

This is a rebuild of a course project. The original backtest let the GA see the next day's prices and credited only
that day, so its headline Sharpe ratio of 1.23 is withdrawn. [What the rebuild fixed](#what-the-rebuild-fixed) lists
each change; the original notebook and paper are kept in [legacy/](legacy/).

## Results

Not run yet. The method below was committed before any backtest on real prices.

**Decision rule, declared before the runs.** RAGA counts as better than a baseline only if the 95% bootstrap
interval of its Sharpe difference against that baseline lies above zero for at least 4 of the 5 GA seeds. The
results will be reported either way.

## What the rebuild fixed

| Original notebook | Now |
|---|---|
| The GA's fitness scored each candidate on the next day's realised return (`prices.iloc[t+1]`). | Expected return is each asset's mean daily return over the two-year training window; a test checks that changing future prices cannot change a decision. |
| The fitness ranked returns in reverse, so it preferred the lowest-return candidates. | Higher expected return ranks higher; a test pins the direction. |
| The backtest credited one day of return per 21-day period, the day after each rebalance. | Holdings drift daily between rebalances and every weekday counts. |
| Sharpe was annualised with √252 on observations 21 days apart. | Annualised on daily returns (261 weekdays a year), in excess of the cash rate. |
| Leverage rose with recent volatility (short over long volatility). | Long over short: exposure falls as recent volatility rises. Leverage pays a 6% cash rate. |
| "HRP" computed the cluster order and then split the assets in ticker order. | HRP as published: correlation clustering, quasi-diagonal order, recursive bisection. |
| Missing prices were back-filled, so ETH had a flat, invented price before its first quote in November 2017. | An asset joins only once it has two years of real prices. |
| A fallback multiplied 5 of the 20 dollar assets by one constant FX ratio. | Every dollar asset is converted with the daily USD/INR rate. |
| Nothing was seeded, and no output file was kept: the paper's figures, including a Sortino ratio and the baselines' Sharpe ratios, are not computed anywhere in the notebook. | Seeded runs, one JSON per strategy and seed, and `python -m raga.report` regenerates every table and the figure. |

## Method

- **Universe and data.** The original 26 assets, from Yahoo Finance adjusted closes, 2015-01-01 to 2026-10-30,
  converted to rupees, on a weekday calendar. Prices are downloaded once and not redistributed; each run records
  their sha256.
- **Decisions.** Every 21 weekdays, from the previous two years only: HRP weights; a volatility veto (average 21-day
  asset volatility above 1.5× the 63-day average) that holds HRP alone at 0.8× leverage; otherwise 12% HRP + 88%
  GA, levered by long over short volatility within 0.8–1.2×.
- **GA.** The original's population (80), generations (50), ensemble (2 runs), tournament selection, crossover,
  mutation and rank-based fitness: expected return, minus CVaR, turnover, first-principal-component exposure and a
  consistency penalty. Its parameters were not tuned.
- **Constraints**, for every strategy: long only, at most 20% per asset; crypto ≤ 5%, commodities ≤ 15%, bonds ≤ 50%.
- **Costs and cash.** 5 bps of traded value; cash borrowed or lent at 6% a year.
- **Baselines.** HRP, equal weight within the same caps, 60/40 (equity ETFs / bonds), the NIFTY 50 ETF bought and held,
  and RAGA's own weights without leverage.
- **Seeds.** The GA runs with 5 seeds; baselines are deterministic.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m raga.run --download      # prices into data/
python -m raga.run --baselines --seeds 0 1 2 3 4
python -m raga.report              # results/RESULTS.md and results/growth.png
pytest -q
```

```
raga/config.py      every setting that changes results
raga/data.py        download, rupee conversion, weekday calendar, eligibility
raga/portfolio.py   constraints and HRP
raga/ga.py          the genetic algorithm and its fitness
raga/backtest.py    strategies and the daily walk-forward simulation
raga/metrics.py     CAGR, volatility, Sharpe, drawdown, block-bootstrap Sharpe differences
raga/report.py      results/runs -> results/RESULTS.md and the figure
legacy/             the original notebook and paper (superseded)
```
