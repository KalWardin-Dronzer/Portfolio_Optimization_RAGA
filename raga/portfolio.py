"""Constraints and Hierarchical Risk Parity.

The original computed HRP's quasi-diagonal order and then bisected the assets in ticker order instead, so its "HRP"
was a variance split of the list as typed. This is HRP as published (Lopez de Prado, 2016): single-linkage
clustering on correlation distance, quasi-diagonalisation, then recursive bisection with inverse-variance cluster
weights.
"""
from __future__ import annotations

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from . import config

_GROUPS = [
    (np.array([config.TICKERS.index(t) for t in tickers]), cap) for tickers, cap in config.GROUP_CAPS.values()
]


def project(weights: np.ndarray, eligible: np.ndarray, max_asset: float = config.MAX_ASSET_WEIGHT) -> np.ndarray:
    """Long-only weights on eligible assets, summing to 1, within the per-asset and group caps.

    Mass removed by a cap goes to the assets not at a cap, in proportion to their weights, until every cap holds.
    """
    w = np.where(eligible, np.clip(np.nan_to_num(weights, nan=0.0), 0.0, None), 0.0)
    if w.sum() <= 0:
        w = eligible.astype(float)
    w = w / w.sum()
    for _ in range(500):
        bound = np.zeros(len(w), dtype=bool)
        for idx, cap in _GROUPS:
            total = w[idx].sum()
            if total > cap:
                w[idx] *= cap / total
            if w[idx].sum() >= cap - 1e-12:
                bound[idx] = True
        w = np.minimum(w, max_asset)
        bound |= w >= max_asset - 1e-12
        deficit = 1.0 - w.sum()
        if deficit < 1e-12:
            break
        free = eligible & ~bound
        if not free.any():
            break  # the caps cannot be met with full investment; the remainder stays in cash
        if w[free].sum() > 0:
            w[free] *= (w[free].sum() + deficit) / w[free].sum()
        else:
            w[free] += deficit / free.sum()
    return w


def hrp(returns_window: np.ndarray) -> np.ndarray:
    """HRP weights for the columns of a (days x assets) return matrix with no missing values."""
    n = returns_window.shape[1]
    if n == 1:
        return np.ones(1)
    cov = np.cov(returns_window, rowvar=False)
    std = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(std, std), -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    order = list(sch.leaves_list(sch.linkage(squareform(dist, checks=False), method="single")))
    weights = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [half for c in clusters if len(c) > 1 for half in (c[: len(c) // 2], c[len(c) // 2:])]
        for left, right in zip(clusters[::2], clusters[1::2]):
            var_left, var_right = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights / weights.sum()


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)
