"""Walk-forward backtest: decide at a rebalance from history only, then hold, day by day, until the next one.

At each rebalance close t a strategy sees prices up to and including t (the training window) and returns target
weights and a leverage. The trade happens at that close and costs COST_BPS of the value traded; from t + 1 the
holdings drift with their own returns, and the cash leg (negative when levered) earns or pays CASH_RATE. Every
weekday is in the return series.

The original credited one day of return per 21-day period, the day after the rebalance, so its equity curve was
made only of the days its fitness had already seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, ga
from .data import eligible, returns
from .portfolio import hrp, project

T = config.TICKERS
DAILY_CASH = (1 + config.CASH_RATE) ** (1 / config.PERIODS_PER_YEAR) - 1


@dataclass
class Decision:
    weights: np.ndarray
    leverage: float = 1.0
    info: dict = field(default_factory=dict)


def rebalance_days(n_days: int) -> list[int]:
    return list(range(config.TRAIN_WINDOW, n_days - 1, config.REBALANCE_EVERY))


def _window(prices: pd.DataFrame, t: int) -> tuple[np.ndarray, np.ndarray]:
    """The eligible assets and their returns over the training window ending at close t (inclusive)."""
    history = prices.iloc[t - config.TRAIN_WINDOW: t + 1]
    ok = eligible(history)
    return ok, returns(history).iloc[1:].to_numpy()[:, ok]


def equal_weight(prices: pd.DataFrame, t: int, state: dict) -> Decision:
    ok, _ = _window(prices, t)
    return Decision(project(ok.astype(float), ok))


def hrp_weights(prices: pd.DataFrame, t: int, state: dict) -> Decision:
    ok, window = _window(prices, t)
    w = np.zeros(len(T))
    w[ok] = hrp(window)
    return Decision(project(w, ok))


def sixty_forty(prices: pd.DataFrame, t: int, state: dict) -> Decision:
    ok, _ = _window(prices, t)
    w = np.zeros(len(T))
    for tickers, share in ((config.SIXTY_FORTY_EQUITY, 0.6), (config.BONDS, 0.4)):
        idx = [T.index(x) for x in tickers if ok[T.index(x)]]
        w[idx] = share / len(idx)
    return Decision(w)


def nifty_buy_hold(prices: pd.DataFrame, t: int, state: dict) -> Decision | None:
    if state.get("bought"):
        return None  # held, never rebalanced
    state["bought"] = True
    w = np.zeros(len(T))
    w[T.index("NIFTYBEES.NS")] = 1.0
    return Decision(w)


def raga(prices: pd.DataFrame, t: int, state: dict) -> Decision:
    """The regime-aware GA: HRP alone at 0.8x in a volatility spike, otherwise 12% HRP + 88% GA, levered by
    long-term over short-term volatility within [0.8, 1.2]."""
    ok, window = _window(prices, t)
    hrp_w = np.zeros(len(T))
    hrp_w[ok] = hrp(window)
    hrp_w = project(hrp_w, ok)
    vol_short = window[-config.VOL_SHORT:].std(axis=0, ddof=1).mean()
    vol_long = window[-config.VOL_LONG:].std(axis=0, ddof=1).mean()
    bad = bool(vol_short > config.VOL_SPIKE_K * vol_long)
    if bad:
        weights, leverage = hrp_w, config.LEV_BAD
    else:
        prev = state.get("prev", np.zeros(len(T)))
        ga_w = ga.optimise(window, prev, hrp_w, ok, state["rng"])
        weights = project(config.BLEND_GOOD * hrp_w + (1 - config.BLEND_GOOD) * ga_w, ok)
        # Lower exposure when recent volatility is high: the original used short over long, the other way round.
        leverage = float(np.clip(vol_long / vol_short, config.LEV_BAD, config.LEV_GOOD))
        leverage = float(np.clip(leverage, config.LEV_MIN, config.LEV_MAX))
    state["prev"] = weights
    return Decision(weights, leverage, {"bad_regime": bad})


STRATEGIES = {"equal_weight": equal_weight, "hrp": hrp_weights, "sixty_forty": sixty_forty,
              "nifty_buy_hold": nifty_buy_hold, "raga": raga}


def decide(strategy, prices: pd.DataFrame, seed: int | None = None) -> dict[int, Decision]:
    """Decisions at every rebalance. Each call sees prices only up to its rebalance day."""
    state: dict = {"rng": np.random.default_rng(seed)}
    decisions = {}
    for t in rebalance_days(len(prices)):
        decision = strategy(prices.iloc[: t + 1], t, state)
        if decision is not None:
            decisions[t] = decision
    return decisions


@dataclass
class Book:
    returns: pd.Series
    turnover: float
    rebalances: int


def simulate(prices: pd.DataFrame, decisions: dict[int, Decision], leverage: bool = True) -> Book:
    """Daily net returns from the day after the first rebalance to the end. `leverage=False` holds the same
    weights unlevered."""
    asset = returns(prices).fillna(0.0).to_numpy()
    days = sorted(decisions)
    exposure = np.zeros(len(T))
    cash = 1.0
    out, turnover = [], 0.0
    for i in range(days[0], len(prices) - 1):
        cost = 0.0
        if i in decisions:
            d = decisions[i]
            target = d.weights * (d.leverage if leverage else 1.0)
            traded = float(np.abs(target - exposure).sum())
            turnover += traded
            cost = traded * config.COST_BPS / 1e4
            exposure, cash = target, 1.0 - target.sum()
        r = asset[i + 1]
        gross = float(exposure @ r + cash * DAILY_CASH)
        out.append((1 - cost) * (1 + gross) - 1)
        exposure = exposure * (1 + r) / (1 + gross)
        cash = cash * (1 + DAILY_CASH) / (1 + gross)
    index = prices.index[days[0] + 1:]
    return Book(pd.Series(out, index=index), turnover, len(days))
