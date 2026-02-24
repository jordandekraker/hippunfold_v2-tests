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
from matplotlib.patches import Patch


# -------------------------
# Fixed settings (per your request)
# -------------------------

FEATS = ["thickness", "SA", "curv", "morph"]
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

def _metric_title(metric: str) -> str:
    return "Accuracy" if metric == "accuracies" else "ROC AUC"


def _annotate_sig(ax, x1: float, x2: float, y: float, text: str):
    if not text:
        return
    h = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1, color='k')
    ax.text((x1 + x2) / 2.0, y + h * 1.2, text, ha="center", va="bottom", fontsize=11)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def _scatter_jitter(ax, data: np.ndarray, pos: float, seed: int):
    if data.size == 0:
        return
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=pos, scale=0.015, size=data.size)
    ax.scatter(
        x, data,
        s=14,
        alpha=0.35,
        color="#7f7f7f",
        edgecolors="none",
        zorder=1,
    )


def _mean_and_err(data: np.ndarray):
    """Return (mean, SD)."""
    m = float(np.nanmean(data))
    if data.size <= 1:
        return m, 0.0
    sd = float(np.nanstd(data, ddof=1))
    return m, sd


def _plot_mean_dot_with_error(ax, data: np.ndarray, pos: float, color: str):
    """Large mean dot + vertical error bar."""
    if data.size == 0:
        return
    m, e = _mean_and_err(data)
    ax.errorbar(
        [pos], [m],
        yerr=[e],
        fmt="o",
        markersize=9.5,          # large mean dot
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        ecolor=color,
        elinewidth=2.0,
        capsize=4,
        capthick=2.0,
        zorder=3,
    )


def plot_panel(
    data: Dict[Key, np.ndarray],
    metric: str,
    outpath: str,
    show: bool = False,
):
    """
    Single figure for a metric (Accuracy or ROC AUC), fixed split=test, fixed features list.
    For each feature: LR and SVM groups side-by-side; within each: OLD and NEW.
    Adds paired permutation-test stars for OLD vs NEW within LR, and within SVM.
    """
    fig = plt.figure(figsize=(max(10, 1.25 * len(FEATS)), 6))
    ax = fig.add_subplot(1, 1, 1)

    # positions
    group_sep = 0.36   # LR vs SVM within feature
    within_sep = 0.14  # OLD vs NEW within classifier group

    # Collect y-lims across the fixed set
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
    ymin = 0
    ymax = 1
    yr = ymax - ymin
    pad = 0.08 * (yr if yr > 0 else 1.0)
    ax.set_ylim(ymin - pad, ymax + 3 * pad)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Mean dot colors (OLD vs NEW)
    mean_color = {"OLD": "#1f77b4", "NEW": "#ff7f0e"}  # blue / orange

    xticks: List[float] = []
    xlabels: List[str] = []

    for i, feat in enumerate(FEATS):
        center = float(i)
        xticks.append(center)
        xlabels.append(feat)

        for clf in CLFS:
            group_center = center - (group_sep / 2.0) if clf == "LR" else center + (group_sep / 2.0)
            pos_old = group_center - (within_sep / 2.0)
            pos_new = group_center + (within_sep / 2.0)

            old = data.get(Key(feat, "OLD", clf, SPLIT, metric), np.array([]))
            new = data.get(Key(feat, "NEW", clf, SPLIT, metric), np.array([]))

            # jittered points
            if old.size:
                _scatter_jitter(ax, old, pos_old, seed=1000 + i + (0 if clf == "LR" else 50))
                _plot_mean_dot_with_error(ax, old, pos_old, color=mean_color["OLD"])

            if new.size:
                _scatter_jitter(ax, new, pos_new, seed=2000 + i + (0 if clf == "LR" else 50))
                _plot_mean_dot_with_error(ax, new, pos_new, color=mean_color["NEW"])

            # significance stars (paired OLD vs NEW)
            if old.size and new.size:
                a2, b2 = _paired(old, new)
                seed_local = (
                    PERM_SEED
                    + 10_000 * (0 if metric == "accuracies" else 1)
                    + 1_000 * i
                    + (0 if clf == "LR" else 100)
                )
                p = _paired_permutation_pvalue(a2, b2, n_perm=N_PERMUTATIONS, seed=seed_local, two_sided=True)
                stars = _p_to_stars(p)

                local_max = float(np.nanmax(np.concatenate([a2, b2]))) if a2.size else float(np.nanmax(np.concatenate([old, new])))
                y = local_max + pad * (1.2 if clf == "LR" else 2.0)  # stagger
                _annotate_sig(ax, pos_old, pos_new, y, stars)

        # classifier labels near bottom
        ax.text(center - (group_sep / 2.0),
                ax.get_ylim()[0] + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "LR", ha="center", va="bottom", fontsize=9)
        ax.text(center + (group_sep / 2.0),
                ax.get_ylim()[0] + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "SVM", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_ylabel(_metric_title(metric))
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Test {_metric_title(metric)} | OLD={VER_LABEL['OLD']} vs NEW={VER_LABEL['NEW']} | "
        f"Paired permutation test (sign-flip, n_perm={N_PERMUTATIONS}) OLD vs NEW within LR/SVM "
        f"(* p<0.05, ** p<0.01, *** p<0.001)"
    )

    # Legend (OLD vs NEW mean dot)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=mean_color["OLD"], markeredgecolor="black",
               markersize=9, label=f"v1.4.1 mean ± SD"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=mean_color["NEW"], markeredgecolor="black",
               markersize=9, label=f"v2.0.0 mean ± SD"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1, 0.05),  # move up slightly
        frameon=True
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
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
                        p_paired_perm=float(p),
                        n_perm=int(N_PERMUTATIONS),
                        sig=_p_to_stars(p),
                    )
                )
    if rows:
        df = pd.DataFrame(rows).sort_values(["metric", "feat", "clf"])
        out_csv = os.path.join(args.outdir, "summary_permtest_fixed_feats_test.csv")
        df.to_csv(out_csv, index=False)
        print(f"[WROTE] {out_csv}")


if __name__ == "__main__":
    main()
