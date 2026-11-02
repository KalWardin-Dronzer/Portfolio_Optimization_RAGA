"""Performance from daily net returns. Annualised with 261 weekdays a year; Sharpe in excess of the cash rate."""
from __future__ import annotations

import numpy as np

from . import config

P = config.PERIODS_PER_YEAR
DAILY_CASH = (1 + config.CASH_RATE) ** (1 / P) - 1


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    return float((r.mean() - DAILY_CASH) / r.std(ddof=1) * np.sqrt(P))


def max_drawdown(r: np.ndarray) -> float:
    wealth = np.cumprod(1 + np.asarray(r, float))
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def summary(r: np.ndarray, turnover: float = 0.0) -> dict[str, float]:
    r = np.asarray(r, float)
    years = len(r) / P
    return {
        "cagr": float(np.prod(1 + r) ** (1 / years) - 1),
        "volatility": float(r.std(ddof=1) * np.sqrt(P)),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "turnover_per_year": turnover / years,
    }


def block_bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, block: int = 21, n_boot: int = 2000,
                                seed: int = 0) -> tuple[float, float, float]:
    """Sharpe(a) - Sharpe(b) on the same days, with a 95% moving-block bootstrap interval (blocks keep a month of
    autocorrelation and resample both series together)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    starts_needed = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n - block + 1, starts_needed)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        diffs[k] = sharpe(a[idx]) - sharpe(b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return sharpe(a) - sharpe(b), float(lo), float(hi)
