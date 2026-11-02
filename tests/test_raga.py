from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raga import backtest, config, data, ga, metrics
from raga.portfolio import hrp, project

T = config.TICKERS


def _prices(days: int = 700, seed: int = 0, late: str | None = "ETH-USD") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=days)
    vol = rng.uniform(0.005, 0.03, len(T))
    rets = rng.normal(0.0003, vol, (days, len(T)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=index, columns=T)
    if late:
        prices.iloc[:200, T.index(late)] = np.nan  # listed later
    return prices


def test_future_prices_cannot_change_a_decision() -> None:
    prices = _prices()
    t = backtest.rebalance_days(len(prices))[0]
    changed = prices.copy()
    changed.iloc[t + 1:] *= np.random.default_rng(9).uniform(0.5, 1.5, changed.iloc[t + 1:].shape)
    for name, strategy in backtest.STRATEGIES.items():
        a = strategy(prices.iloc[: t + 1], t, {"rng": np.random.default_rng(3)})
        b = strategy(changed.iloc[: t + 1], t, {"rng": np.random.default_rng(3)})
        assert np.array_equal(a.weights, b.weights) and a.leverage == b.leverage, name
    # And the full decision sequence only ever reads the past: identical up to the last rebalance before a change.
    cut = backtest.rebalance_days(len(prices))[1]
    later = prices.copy()
    later.iloc[cut + 1:] *= 1.3
    first = backtest.decide(backtest.raga, prices, seed=1)
    second = backtest.decide(backtest.raga, later, seed=1)
    for day in (d for d in first if d <= cut):
        assert np.array_equal(first[day].weights, second[day].weights)


def test_fitness_prefers_the_higher_expected_return() -> None:
    rng = np.random.default_rng(0)
    window = rng.normal(0, 0.01, (300, 4))
    window[:, 0] += 0.002  # asset 0 has the highest mean return
    eligible = np.zeros(len(T), dtype=bool)
    eligible[[0, 1, 2, 3]] = True
    fitness = ga.Fitness(window, eligible, prev_weights=np.zeros(len(T)))
    high, low = np.zeros(len(T)), np.zeros(len(T))
    high[[0, 1]] = 0.5
    low[[1, 2]] = 0.5
    ret = fitness.components(np.array([high, low]))["ret"]
    assert ret[0] > ret[1]
    ranks = (ga._ranks(ret) + 1) / 2
    assert ranks[0] > ranks[1]  # the original's (n - rank) / n ranked these the other way round


def test_projection_meets_every_cap() -> None:
    rng = np.random.default_rng(1)
    eligible = np.ones(len(T), dtype=bool)
    for _ in range(200):
        w = project(rng.random(len(T)) ** 4, eligible)
        assert w.sum() == pytest.approx(1.0) and w.min() >= 0
        assert w.max() <= config.MAX_ASSET_WEIGHT + 1e-9
        for tickers, cap in config.GROUP_CAPS.values():
            assert w[[T.index(x) for x in tickers]].sum() <= cap + 1e-9
    partial = eligible.copy()
    partial[T.index("ETH-USD")] = False
    assert project(np.ones(len(T)), partial)[T.index("ETH-USD")] == 0


def test_hrp_is_inverse_variance_for_uncorrelated_assets() -> None:
    rng = np.random.default_rng(2)
    window = rng.normal(0, 1, (20000, 2)) * np.array([0.01, 0.02])
    w = hrp(window)
    assert w == pytest.approx(np.array([0.8, 0.2]), abs=0.01)


def test_hrp_uses_the_cluster_order_not_the_ticker_order() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(0, 0.01, (5000, 2))
    noise = rng.normal(0, 0.002, (5000, 4))
    # columns 0 and 2 move together, 1 and 3 move together: listed interleaved
    window = np.column_stack([base[:, 0], base[:, 1], base[:, 0], base[:, 1]]) + noise
    w = hrp(window)
    assert w.sum() == pytest.approx(1.0)
    assert w == pytest.approx(np.full(4, 0.25), abs=0.03)


def test_leverage_falls_when_recent_volatility_rises() -> None:
    prices = _prices(late=None)
    t = backtest.rebalance_days(len(prices))[0]
    calm, stormy = prices.iloc[: t + 1].copy(), prices.iloc[: t + 1].copy()
    rets = data.returns(stormy).iloc[1:]
    rets.iloc[-15:] *= 1.4  # recent volatility up, below the 1.5x veto
    stormy.iloc[1:] = 100 * (1 + rets.fillna(0)).cumprod().to_numpy()
    lev_calm = backtest.raga(calm, t, {"rng": np.random.default_rng(0)}).leverage
    lev_stormy = backtest.raga(stormy, t, {"rng": np.random.default_rng(0)}).leverage
    assert lev_stormy < lev_calm


def test_holdings_drift_between_rebalances_and_every_day_counts() -> None:
    prices = _prices(late=None)
    days = backtest.rebalance_days(len(prices))
    w = np.zeros(len(T))
    w[[0, 1]] = 0.5
    book = backtest.simulate(prices, {days[0]: backtest.Decision(w)})
    assert len(book.returns) == len(prices) - days[0] - 1
    held = prices.iloc[days[0]:, [0, 1]]
    value = (held / held.iloc[0]).mean(axis=1)
    expected = np.array((value / value.shift(1) - 1).iloc[1:], dtype=float)
    expected[0] = (1 - 1.0 * config.COST_BPS / 1e4) * (1 + expected[0]) - 1
    assert book.returns.to_numpy() == pytest.approx(expected, abs=1e-12)


def test_leverage_pays_the_cash_rate() -> None:
    prices = _prices(late=None)
    prices.iloc[:, :] = 100.0  # no asset moves
    days = backtest.rebalance_days(len(prices))
    w = np.zeros(len(T))
    w[0] = 1.0
    book = backtest.simulate(prices, {days[0]: backtest.Decision(w, leverage=1.5)})
    # the borrowed half pays the cash rate; the debt compounds, so later days cost a little more
    assert book.returns.iloc[1] == pytest.approx(-0.5 * backtest.DAILY_CASH, rel=1e-2)
    assert book.returns.iloc[-1] < book.returns.iloc[1] < 0


def test_no_back_filling_before_the_first_quote() -> None:
    raw = pd.DataFrame(
        {t: [10.0, 11.0, 12.0] for t in T} | {config.FX_TICKER: [80.0, 81.0, 82.0]},
        index=pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"]),
    )
    raw.loc["2020-01-06", "ETH-USD"] = np.nan
    prices = data.to_inr_weekdays(raw)
    assert np.isnan(prices.loc["2020-01-06", "ETH-USD"])
    assert prices.loc["2020-01-07", "SPY"] == pytest.approx(11.0 * 81.0)  # dollar assets in rupees
    assert prices.loc["2020-01-07", "TCS.NS"] == pytest.approx(11.0)


def test_sharpe_annualisation() -> None:
    r = np.random.default_rng(5).normal(0.0005, 0.01, 5000)
    expected = (r.mean() - metrics.DAILY_CASH) / r.std(ddof=1) * np.sqrt(config.PERIODS_PER_YEAR)
    assert metrics.sharpe(r) == pytest.approx(expected)
    diff, lo, hi = metrics.block_bootstrap_sharpe_diff(r, r, n_boot=50)
    assert diff == 0 and lo == pytest.approx(0) and hi == pytest.approx(0)
