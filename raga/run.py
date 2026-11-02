"""Run every strategy and write one JSON per strategy and seed to results/runs/.

    python -m raga.run --download          fetch prices into data/ (Yahoo Finance; not redistributed)
    python -m raga.run --seeds 0 1 2 3 4   baselines once, the GA once per seed
    python -m raga.report                  results/RESULTS.md and the figure
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from . import backtest, config, data, metrics

RUNS = data.ROOT / "results" / "runs"


def _commit() -> str:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=data.ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=data.ROOT, capture_output=True,
                           text=True).stdout.strip()
    return f"{head}+uncommitted-changes" if dirty else head


def _write(name: str, seed: int | None, book: backtest.Book, decisions: dict, prices, digest: str, seconds: float) -> None:
    infos = [d.info for d in decisions.values()]
    record = {
        "strategy": name,
        "seed": seed,
        "commit": _commit(),
        "data": {"sha256": digest, "start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
                 "first_return": str(book.returns.index[0].date()), "days": len(book.returns)},
        "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
        "rebalances": book.rebalances,
        "bad_regime_share": float(np.mean([i["bad_regime"] for i in infos])) if infos and "bad_regime" in infos[0] else None,
        "mean_leverage": float(np.mean([d.leverage for d in decisions.values()])) if name == "raga" else 1.0,
        "metrics": metrics.summary(book.returns.to_numpy(), book.turnover),
        "seconds": seconds,
        "python": platform.python_version(),
        "dates": [str(d.date()) for d in book.returns.index],
        "returns": [round(float(x), 10) for x in book.returns],
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / (f"{name}_seed{seed}.json" if seed is not None else f"{name}.json")
    path.write_text(json.dumps(record), encoding="utf-8")
    m = record["metrics"]
    print(f"{path.name}: CAGR {m['cagr']:.2%}, vol {m['volatility']:.2%}, Sharpe {m['sharpe']:.2f}, "
          f"max drawdown {m['max_drawdown']:.2%}, {seconds:.0f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("--baselines", action="store_true", help="run the four baselines")
    args = parser.parse_args(argv)
    if args.download:
        print(f"wrote {data.download()} (sha256 {data.sha256()[:12]})")
    if not args.seeds and not args.baselines:
        return
    prices, digest = data.load(), data.sha256()
    if args.baselines:
        for name in ("equal_weight", "hrp", "sixty_forty", "nifty_buy_hold"):
            started = time.perf_counter()
            decisions = backtest.decide(backtest.STRATEGIES[name], prices)
            _write(name, None, backtest.simulate(prices, decisions), decisions, prices, digest, time.perf_counter() - started)
    for seed in args.seeds:
        started = time.perf_counter()
        decisions = backtest.decide(backtest.raga, prices, seed=seed)
        seconds = time.perf_counter() - started
        _write("raga", seed, backtest.simulate(prices, decisions), decisions, prices, digest, seconds)
        _write("raga_unlevered", seed, backtest.simulate(prices, decisions, leverage=False), decisions, prices, digest, seconds)


if __name__ == "__main__":
    main()
