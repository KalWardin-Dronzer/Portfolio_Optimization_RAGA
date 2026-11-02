"""Prices: download once from Yahoo Finance, cache, convert to rupees, put on a weekday calendar.

The original back-filled missing prices, so ETH (first quoted November 2017) carried a flat, invented price for
almost three years, and its fallback FX was one constant ratio applied to five of the 20 dollar assets. Here an
asset has no price before its first quote and is simply not eligible until it has a full training window, and every
dollar asset is converted with the daily USD/INR rate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "prices_raw.csv"


def download(path: Path = CACHE) -> Path:
    import yfinance as yf

    tickers = config.TICKERS + [config.FX_TICKER]
    raw = yf.download(tickers, start=config.START, end=config.END, auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    close = pd.concat({t: raw[t]["Close"] for t in tickers}, axis=1)
    missing = [t for t in tickers if close[t].dropna().empty]
    if missing:
        raise RuntimeError(f"Yahoo returned no prices for {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    close.sort_index().to_csv(path, float_format="%.8g")
    return path


def sha256(path: Path = CACHE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def drop_bad_prints(raw: pd.DataFrame, jump: float = 0.25, back: float = 0.05, max_days: int = 3) -> tuple[pd.DataFrame, list]:
    """Remove quotes that jump more than `jump` and come back to within `back` of the last good price within
    `max_days` quotes: data errors, not market moves. Yahoo has two in this window: NIFTYBEES at about a tenth of its
    price on 19-20 December 2019, and CPER for one day in February 2015. A real crash does not come back."""
    raw = raw.copy()
    removed = []
    for column in raw.columns:
        s = raw[column].dropna()
        values, i = s.to_numpy(), 1
        while i < len(values):
            if abs(values[i] / values[i - 1] - 1) > jump:
                for j in range(1, max_days + 1):
                    if i + j < len(values) and abs(values[i + j] / values[i - 1] - 1) < back:
                        removed += [(column, str(d.date())) for d in s.index[i: i + j]]
                        raw.loc[s.index[i: i + j], column] = np.nan
                        i += j
                        break
            i += 1
    return raw, removed


def to_inr_weekdays(raw: pd.DataFrame) -> pd.DataFrame:
    """Weekday calendar, each series carried forward from its own last quote (never backward), dollar assets
    converted to rupees with that day's USD/INR."""
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index)
    raw, _ = drop_bad_prints(raw)
    days = pd.bdate_range(raw.index.min(), raw.index.max())
    weekday = raw.reindex(raw.index.union(days)).ffill().reindex(days)
    fx = weekday[config.FX_TICKER]
    if fx.isna().any():
        first = fx.first_valid_index()
        weekday = weekday.loc[first:]
        fx = fx.loc[first:]
    prices = weekday[config.TICKERS].copy()
    dollar = [t for t in config.TICKERS if t not in config.INDIA]
    prices[dollar] = prices[dollar].mul(fx, axis=0)
    # Start on the first day every asset quoted from the outset has a price. The download starts on 1 January, a
    # holiday in some markets; starting there made a window "missing" a day for those assets, which then counted
    # as not yet listed at the first rebalance.
    firsts = prices.apply(lambda s: s.first_valid_index())
    early = firsts[firsts <= prices.index[0] + pd.Timedelta(days=30)]
    return prices.loc[early.max():]


def load(path: Path = CACHE) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m raga.run --download` first")
    return to_inr_weekdays(pd.read_csv(path, index_col=0))


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; NaN before an asset's first quote."""
    return prices / prices.shift(1) - 1.0


def eligible(prices_window: pd.DataFrame) -> np.ndarray:
    """Assets quoted on every day of the window: an asset joins only once it has a full history behind it."""
    return prices_window.notna().all(axis=0).to_numpy()
