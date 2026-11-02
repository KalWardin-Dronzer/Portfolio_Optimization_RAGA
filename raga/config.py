"""Every setting that changes results, in one place.

The universe, the GA's parameters and fitness weights, the group caps, the volatility veto, the blend and the
leverage bounds are the original notebook's values (legacy/RAGA_Code.ipynb), unchanged and untuned. What changed is
listed in the README under "What the rebuild fixed".
"""
from __future__ import annotations

# The original 26-asset universe, by group.
INDIA = ["NIFTYBEES.NS", "HDFCBANK.NS", "RELIANCE.NS", "INFY.NS", "TCS.NS", "ASIANPAINT.NS"]
US_EQUITY = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLY", "SMH", "ARKK"]
GLOBAL_EQUITY = ["EEM", "VGK", "EWJ"]
COMMODITIES = ["GLD", "SLV", "USO", "CPER"]
BONDS = ["TLT", "IEF", "SHY"]
CRYPTO = ["BTC-USD", "ETH-USD"]
TICKERS = INDIA + US_EQUITY + GLOBAL_EQUITY + COMMODITIES + BONDS + CRYPTO
FX_TICKER = "USDINR=X"  # every non-Indian asset is priced in US dollars and converted to rupees with it

# Data window: pinned, so a rerun downloads the same period (Yahoo may still revise adjusted closes; each run
# records the sha256 of the prices it used).
START = "2015-01-01"
END = "2026-10-31"  # exclusive

# Calendar: weekdays. Crypto's weekend moves land in Monday's return, and an exchange holiday is a zero-return
# day for the assets that did not trade. 261 weekdays a year.
PERIODS_PER_YEAR = 261
TRAIN_WINDOW = 522  # two years of weekdays of history behind every decision
REBALANCE_EVERY = 21  # about monthly
VOL_SHORT, VOL_LONG = 21, 63

# Costs and cash: 5 bps of traded value at each rebalance (the original's TC_BPS), and cash, borrowed or lent,
# at a constant 6% a year, roughly the Indian T-bill rate over the period. Sharpe ratios are in excess of it.
COST_BPS = 5.0
CASH_RATE = 0.06

# Constraints, applied to every strategy alike.
GROUP_CAPS = {"crypto": (CRYPTO, 0.05), "commodities": (COMMODITIES, 0.15), "bonds": (BONDS, 0.50)}
MAX_ASSET_WEIGHT = 0.20

# GA (original values).
POP_SIZE = 80
N_GENERATIONS = 50
ENSEMBLE_RUNS = 2
TOURNAMENT_K = 3
ELITISM = 2
MUTATION_RATE = 0.2
MUTATION_STD = 0.03
W_RET, W_RISK, W_TURN, W_PCA = 1.0, 1.0, 0.7, 0.5
CONSISTENCY_PENALTY = 0.8
CVAR_ALPHA = 0.05

# Regime rule and leverage (original values).
VOL_SPIKE_K = 1.5  # "bad" regime: average 21-day asset volatility above 1.5x the 63-day average
BLEND_GOOD, BLEND_BAD = 0.12, 1.0  # share of HRP in the final weights
LEV_BAD, LEV_GOOD = 0.8, 1.2
LEV_MIN, LEV_MAX = 0.5, 1.5

# 60/40 baseline: the original's equity ETFs and its bonds.
SIXTY_FORTY_EQUITY = ["SPY", "QQQ", "IWM", "EEM", "VGK", "EWJ", "ARKK", "SMH", "XLY", "XLK", "XLF"]
