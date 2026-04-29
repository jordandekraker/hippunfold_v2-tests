#!/usr/bin/env python3
"""
plot_hippunfold_metrics_panel.py

Reads only:
  *_test_accuracies.csv
  *_test_roc_aucs.csv

Usage:
  python TLEclass.py /host/verges/tank/data/hippunfold_comparison/LAT --outdir ./plots_TLE
"""

from __future__ import annotations

import argparse
import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Fixed settings (per your request)
# -------------------------

FEATS = ["thickness", "curv"]
SPLIT = "test"
CLFS = ["LR", "SVM"]
METRICS = ["accuracies", "roc_aucs"]  # -> Accuracy, ROC AUC

# -------------------------
# Permutation test settings
# -------------------------
N_PERMUTATIONS = 10000  # bump this up later (e.g., 10_000 or 100_000)
PERM_SEED = 1337      # for reproducibility


# -------------------------
# Parsing / conventions
# -------------------------

FNAME_RE = re.compile(
    r"""
    ^(?P<prefix>.+?)_               # dataset prefix
    feat-(?P<feat>[^_]+)_           # feat-XYZ
    (?P<ver>OLD|NEW)_               # OLD/NEW
    (?P<clf>LR|SVM)_                # LR/SVM
    (?P<split>test_)?               # optional test_
    (?P<metric>accuracies|roc_aucs) # metric
    \.csv$
    """,
    re.VERBOSE,
)

VER_LABEL = {"OLD": "hippunfold_v1.4.1", "NEW": "hippunfold_v2.0.0"}


@dataclass(frozen=True)
class Key:
    feat: str
    ver: str       # OLD/NEW
    clf: str       # LR/SVM
    split: str     # test
    metric: str    # accuracies/roc_aucs


def _read_metric_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        s = df[num_cols[0]]
    else:
        s = df[df.columns[0]]
    return pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)


def _discover_files(root: str) -> Dict[Key, str]:
    root = os.path.abspath(root)
    paths = glob.glob(os.path.join(root, "*.csv"))
    out: Dict[Key, str] = {}
    for p in paths:
        base = os.path.basename(p)
        m = FNAME_RE.match(base)
        if not m:
            continue
        feat = m.group("feat")
        ver = m.group("ver")
        clf = m.group("clf")
        split = "test" if m.group("split") else "train"
        metric = m.group("metric")
        out[Key(feat, ver, clf, split, metric)] = p
    return out


def _paired(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n == 0:
        return np.array([]), np.array([])
    return a[:n], b[:n]


# -------------------------
# Paired permutation test (sign-flip / swap within pairs)
# -------------------------

def _paired_permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = N_PERMUTATIONS,
    seed: int = PERM_SEED,
    two_sided: bool = True,
) -> float:
    """
    Paired permutation (randomization) test for difference in means.

    Null: within each paired observation i, labels A/B are exchangeable.
    Implementation: sign-flip the paired differences d_i = b_i - a_i.

    Returns a Monte-Carlo p-value with +1 correction:
      p = (1 + #{|T_perm| >= |T_obs|}) / (1 + n_perm)  (two-sided)
    """
    a2, b2 = _paired(a, b)
    n = a2.size
    if n < 2:
        return 1.0

    d = b2 - a2
    if not np.all(np.isfinite(d)):
        d = d[np.isfinite(d)]
        n = d.size
        if n < 2:
            return 1.0

    t_obs = float(np.mean(d))
    rng = np.random.default_rng(seed)

    # Generate random sign flips: shape (n_perm, n), entries in {-1, +1}
    # Use integers then map to {-1, +1} for speed/clarity.
    flips = rng.integers(0, 2, size=(n_perm, n), dtype=np.int8)
    signs = (2 * flips - 1).astype(np.int8)  # 0->-1, 1->+1

    t_perm = (signs * d[None, :]).mean(axis=1)

    if two_sided:
        extreme = np.sum(np.abs(t_perm) >= abs(t_obs))
    else:
        # one-sided: H1 mean(d) > 0
        extreme = np.sum(t_perm >= t_obs)

    p = (1.0 + float(extreme)) / (1.0 + float(n_perm))
    return float(max(min(p, 1.0), 0.0))

def cohens_d_paired(a, b):
    """
    Compute paired Cohen's d (dz).

    Parameters
    ----------
    a, b : array-like
        Paired observations (same length).

    Returns
    -------
    d_z : float
        Cohen's d for paired samples (mean difference / SD of differences).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Inputs must have the same shape for paired comparison.")

    # Drop NaNs pairwise
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        raise ValueError("Not enough valid paired observations.")

    diff = b - a
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    if sd_diff == 0:
        return np.nan  # undefined

    return mean_diff / sd_diff

def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# -------------------------
# Plot helpers
# -------------------------
# Replace the old plotting helper section with this

from matplotlib.lines import Line2D

RIDGE_COLOR = {"OLD": "#8a8a8a", "NEW": "#d62728"}  # grey / red

def _metric_title(metric: str) -> str:
    return "Accuracy" if metric == "accuracies" else "ROC AUC"

def _kde_1d(x: np.ndarray, grid: np.ndarray, bw: float | None = None) -> np.ndarray:
    """Simple Gaussian KDE to avoid adding scipy/seaborn dependency."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 2:
        return np.zeros_like(grid)

    if bw is None:
        sd = np.std(x, ddof=1)
        bw = 1.06 * sd * (x.size ** (-1 / 5))
        if not np.isfinite(bw) or bw <= 0:
            bw = 0.02

    z = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * z**2).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))
    return dens


def _plot_ridge(ax, data: np.ndarray, y0: float, color: str, grid: np.ndarray, height: float = 0.34):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return

    dens = _kde_1d(data, grid)
    if dens.max() > 0:
        dens = dens / dens.max() * height

    ax.fill_between(grid, y0, y0 + dens, color=color, alpha=0.45, linewidth=0)
    ax.plot(grid, y0 + dens, color=color, linewidth=1.5)

    # median marker
    med = float(np.median(data))
    ax.plot([med, med], [y0, y0 + height * 0.85], color=color, linewidth=1.2)


def plot_panel(
    data: Dict[Key, np.ndarray],
    metric: str,
    outpath: str,
    show: bool = False,
):
    """
    Ridge-plot version:
    y-axis rows are feat × classifier.
    OLD and NEW bootstrap distributions are overlaid within each row.
    """
    rows = [(feat, clf) for feat in FEATS for clf in CLFS]
    fig_h = max(5, 0.62 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(4, fig_h))

    vals = []
    for feat in FEATS:
        for clf in CLFS:
            for ver in ["OLD", "NEW"]:
                arr = data.get(Key(feat, ver, clf, SPLIT, metric), np.array([]))
                if arr.size:
                    vals.append(arr)

    if not vals:
        raise SystemExit(f"No data found for split={SPLIT}, metric={metric} in requested feats={FEATS}")

    allv = np.concatenate(vals)
    xmin = 0
    xmax = 1
    grid = np.linspace(xmin, xmax, 400)

    y_positions = np.arange(len(rows))[::-1]

    for y0, (feat, clf) in zip(y_positions, rows):
        old = data.get(Key(feat, "OLD", clf, SPLIT, metric), np.array([]))
        new = data.get(Key(feat, "NEW", clf, SPLIT, metric), np.array([]))

        _plot_ridge(ax, old, y0, RIDGE_COLOR["OLD"], grid)
        _plot_ridge(ax, new, y0, RIDGE_COLOR["NEW"], grid)

        if old.size and new.size:
            a2, b2 = _paired(old, new)
            seed_local = (
                PERM_SEED
                + 10_000 * (0 if metric == "accuracies" else 1)
                + 1_000 * FEATS.index(feat)
                + (0 if clf == "LR" else 100)
            )
            p = _paired_permutation_pvalue(
                a2, b2, n_perm=N_PERMUTATIONS, seed=seed_local, two_sided=True
            )
            stars = _p_to_stars(p)
            if stars:
                ax.text(
                    xmax,
                    y0 + 0.18,
                    stars,
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{feat} | {clf}" for feat, clf in rows])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.35, len(rows) - 0.25)
    ax.set_xlabel(_metric_title(metric))
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(outpath, dpi=250)
    print(f"[WROTE] {outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Directory containing the CSVs (e.g. .../LAT)")
    ap.add_argument("--outdir", default="plots_panel", help="Output directory")
    ap.add_argument("--show", action="store_true", help="Show figures interactively")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = _discover_files(args.root)
    if not files:
        raise SystemExit(f"No matching metric CSVs found in: {args.root}")

    # load arrays only for requested feats + split=test + metrics of interest
    data: Dict[Key, np.ndarray] = {}
    missing = []
    for feat in FEATS:
        for clf in CLFS:
            for ver in ["OLD", "NEW"]:
                for metric in METRICS:
                    k = Key(feat, ver, clf, SPLIT, metric)
                    p = files.get(k)
                    if p is None:
                        missing.append(k)
                        continue
                    try:
                        data[k] = _read_metric_csv(p)
                    except Exception as e:
                        print(f"[WARN] failed reading {p}: {e}")

    if missing:
        print("[WARN] Some expected files were not found (showing up to 12):")
        for k in missing[:12]:
            print(f"  feat={k.feat} ver={k.ver} clf={k.clf} split={k.split} metric={k.metric}")
        if len(missing) > 12:
            print(f"  ... and {len(missing) - 12} more")

    # Plot two panels: accuracy + AUC
    for metric in METRICS:
        out_png = os.path.join(args.outdir, f"panel_split-{SPLIT}_metric-{metric}.png")
        plot_panel(data, metric, out_png, show=args.show)

    # Write a compact summary CSV (paired permutation tests)
    rows = []
    for feat in FEATS:
        for clf in CLFS:
            for metric in METRICS:
                old = data.get(Key(feat, "OLD", clf, SPLIT, metric), np.array([]))
                new = data.get(Key(feat, "NEW", clf, SPLIT, metric), np.array([]))
                a2, b2 = _paired(old, new)
                if a2.size < 2:
                    continue

                seed_local = (
                    PERM_SEED
                    + 10_000 * (0 if metric == "accuracies" else 1)
                    + 1_000 * FEATS.index(feat)
                    + (0 if clf == "LR" else 100)
                    + 7
                )
                p = _paired_permutation_pvalue(a2, b2, n_perm=N_PERMUTATIONS, seed=seed_local, two_sided=True)
                d = cohens_d_paired(a2,b2)

                rows.append(
                    dict(
                        feat=feat,
                        clf=clf,
                        metric=metric,
                        split=SPLIT,
                        n=int(a2.size),
                        old_mean=float(np.mean(a2)),
                        new_mean=float(np.mean(b2)),
                        diff_mean=float(np.mean(b2 - a2)),
                        cohens_dz=float(d),
                        p_paired_perm=float(p),
                        n_perm=int(N_PERMUTATIONS),
                        sig=_p_to_stars(p),
                    )
                )
    if rows:
        df = pd.DataFrame(rows)
        out_csv = os.path.join(args.outdir, "summary_permtest_fixed_feats_test.csv")
        df.to_csv(out_csv, index=False)
        print(f"[WROTE] {out_csv}")


if __name__ == "__main__":
    main()
