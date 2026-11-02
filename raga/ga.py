"""The genetic algorithm: the original's operators, population and rank-based fitness, with two fixes.

1. Expected return comes from the training window (each asset's mean daily return over it). The original scored
   every candidate on the next day's realised return, the very return the backtest then credited.
2. A higher expected return now ranks higher. The original ranked returns in reverse, so its fitness preferred the
   candidates with the lowest return.

Everything else is unchanged: CVaR, turnover, first-principal-component exposure and a consistency penalty, all
measured against the previous weights; tournament selection, one-point or blend crossover, Gaussian mutation,
elitism, and the mean of an ensemble of runs. Randomness comes from one seeded generator. Candidates are
full-universe vectors with zero weight on assets not yet eligible.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from . import config
from .portfolio import project


def _ranks(values: np.ndarray) -> np.ndarray:
    """0 for the smallest value, n - 1 for the largest."""
    return np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(float)


class Fitness:
    """Scores candidate weight vectors (rows) on one training window of the eligible assets."""

    def __init__(self, returns_window: np.ndarray, eligible: np.ndarray, prev_weights: np.ndarray) -> None:
        n = len(eligible)
        self.returns = np.zeros((returns_window.shape[0], n))
        self.returns[:, eligible] = returns_window
        self.mu = self.returns.mean(axis=0)
        self.pc1 = np.zeros(n)
        self.pc1[eligible] = np.abs(PCA(n_components=1).fit(returns_window).components_[0])
        self.pc1 /= self.pc1.sum()
        self.prev = prev_weights
        self.tail = max(1, int(config.CVAR_ALPHA * returns_window.shape[0]))

    def components(self, pop: np.ndarray) -> dict[str, np.ndarray]:
        series = self.returns @ pop.T
        worst = np.sort(series, axis=0)[: self.tail]
        delta = pop - self.prev
        return {
            "ret": pop @ self.mu,
            "cvar": -worst.mean(axis=0),
            "turnover": np.abs(delta).sum(axis=1),
            "pca": pop @ self.pc1,
            "consistency": (delta ** 2).sum(axis=1),
        }

    def __call__(self, pop: np.ndarray) -> np.ndarray:
        c = self.components(pop)
        n = len(pop)
        return (
            config.W_RET * (_ranks(c["ret"]) + 1) / n
            - config.W_RISK * _ranks(c["cvar"]) / n
            - config.W_TURN * _ranks(c["turnover"]) / n
            - config.W_PCA * _ranks(c["pca"]) / n
            - config.CONSISTENCY_PENALTY * c["consistency"]
        )


def run_once(fitness: Fitness, hrp_w: np.ndarray, eligible: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(eligible)
    fix = lambda w: project(w, eligible)  # noqa: E731
    pop = [fix(fitness.prev + rng.normal(0, 0.02, n)) for _ in range(max(2, int(config.POP_SIZE * 0.10)))]
    pop += [fix(hrp_w + rng.normal(0, 0.01, n)) for _ in range(max(1, int(config.POP_SIZE * 0.05)))]
    while len(pop) < config.POP_SIZE:
        pop.append(fix(rng.random(n)))
    pop = np.array(pop)
    best_w, best_f = pop[0], -np.inf
    for _ in range(config.N_GENERATIONS):
        scores = fitness(pop)
        order = np.argsort(-scores, kind="stable")
        if scores[order[0]] > best_f:
            best_f, best_w = scores[order[0]], pop[order[0]].copy()
        children = [pop[i] for i in order[: config.ELITISM]]
        while len(children) < config.POP_SIZE:
            a = _tournament(pop, scores, rng)
            b = _tournament(pop, scores, rng)
            if rng.random() < 0.5:
                cut = int(rng.integers(1, n))
                child = np.concatenate([a[:cut], b[cut:]])
            else:
                share = rng.random()
                child = share * a + (1 - share) * b
            if rng.random() < config.MUTATION_RATE:
                child = child + rng.normal(0, config.MUTATION_STD, n)
            children.append(fix(child))
        pop = np.array(children)
    return best_w


def _tournament(pop: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    picks = rng.choice(len(pop), size=min(config.TOURNAMENT_K, len(pop)), replace=False)
    return pop[picks[np.argmax(scores[picks])]]


def optimise(returns_window: np.ndarray, prev: np.ndarray, hrp_w: np.ndarray, eligible: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
    """The ensemble mean of GA runs, within the caps. `returns_window` holds the eligible columns only; `prev` and
    `hrp_w` are full-universe vectors."""
    prev = project(prev, eligible) if prev[eligible].sum() > 0 else project(eligible.astype(float), eligible)
    fitness = Fitness(returns_window, eligible, prev)
    runs = [run_once(fitness, hrp_w, eligible, rng) for _ in range(config.ENSEMBLE_RUNS)]
    return project(np.mean(runs, axis=0), eligible)
