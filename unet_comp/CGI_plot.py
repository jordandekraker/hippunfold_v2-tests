#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from scipy.stats import ttest_rel  # <-- NEW (paired t-test)

# ---------------------------
# Config
# ---------------------------
DATASETS   = ["PNI", "MICs"]
CONDITIONS = ["T1w", "synthseg_v0.2", "synthlayer_v0.3"]
JITTER_STD = 0.035
MEAN_S     = 140
DPI        = 160
OUTDIR     = "plots/"
HEMI_MARK  = {"L": "o", "R": "x"}  # hemisphere -> marker

# Use ONLY the 3rd, 4th, 5th colors from the old palette
# old palette = [blue, orange, green, red, purple, brown]
PALETTE_MEANS = [
    "#2ca02c",  # green  (3rd)
    "#d62728",  # red    (4th)
    "#9467bd",  # purple (5th)
]

# T-test comparisons (within each dataset + measure plot):
# synthseg_v0.2 vs T1w, synthlayer_v0.3 vs T1w
TTEST_COMPARISONS = [
    ("T1w", "synthseg_v0.2"),
    ("T1w", "synthlayer_v0.3"),
]

# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser(description="Plot hippunfold reliability & transfer metrics.")
parser.add_argument("--csv", default="metrics_per_subject.csv", help="Input CSV produced by analysis script.")
parser.add_argument("--palette", default="jet", help="Matplotlib colormap name used for subject_row colors.")
args = parser.parse_args()

Path(OUTDIR).mkdir(exist_ok=True)

# ---------------------------
# Load
# ---------------------------
df = pd.read_csv(args.csv)
valid_measures = {"consistency", "identifiability", "generalizability"}
df = df[df["measure"].isin(valid_measures)].copy()

# Ensure types
if "subject_row" in df.columns:
    df["subject_row"] = pd.to_numeric(df["subject_row"], errors="coerce")

# ---------------------------
# Consistent colors for subject_row across all plots
# (kept, though individuals are currently plotted as grey)
# ---------------------------
all_subject_rows = np.unique(df["subject_row"].values)
all_subject_rows = all_subject_rows[~pd.isna(all_subject_rows)]
all_subject_rows = all_subject_rows.astype(int) if len(all_subject_rows) else np.array([], dtype=int)

cmap = plt.get_cmap(args.palette, max(len(all_subject_rows), 1))
color_lookup = {sr: cmap(i % cmap.N) for i, sr in enumerate(sorted(all_subject_rows))}

# ---------------------------
# Helpers: significance stars + bracket annotation
# ---------------------------
def p_to_stars(p: float) -> str | None:
    if not np.isfinite(p):
        return None
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None

def add_sig_bracket(ax, x1, x2, y, text, h=0.02):
    """Draw a bracket from x1->x2 at height y (data coords), and put text centered above."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.4, color="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=12, color="black")

def paired_ttest_by_subject_mean(df_in: pd.DataFrame, value_col: str, cond_a: str, cond_b: str) -> float:
    """
    Paired t-test comparing cond_a vs cond_b using per-subject means across hemispheres.
    Pairs are matched on subject_row. Subjects must have both conditions.
    Returns p-value (nan if insufficient pairs).
    """
    if df_in.empty:
        return float("nan")

    needed = {"subject_row", "condition", "hemi", value_col}
    if not needed.issubset(df_in.columns):
        return float("nan")

    dfa = df_in[df_in["condition"] == cond_a].copy()
    dfb = df_in[df_in["condition"] == cond_b].copy()
    if dfa.empty or dfb.empty:
        return float("nan")

    # mean across hemispheres within each subject
    ma = dfa.groupby("subject_row")[value_col].mean()
    mb = dfb.groupby("subject_row")[value_col].mean()

    common = ma.index.intersection(mb.index)
    if len(common) < 2:
        return float("nan")

    a = ma.loc[common].to_numpy(dtype=float)
    b = mb.loc[common].to_numpy(dtype=float)

    # paired t-test (two-sided)
    res = ttest_rel(a, b, nan_policy="omit")
    return float(res.pvalue)

# ---------------------------
# Jittered scatter helper
# - groups defined by labels & group_cols
# - individuals: grey, shaped by hemisphere
# - mean/SD: colored by label index (using PALETTE_MEANS, cycling if needed)
# - optional paired t-test annotations for condition-only plots
# ---------------------------
def scatter_block(
    df_sub,
    value_col,
    labels,
    group_cols,
    title,
    out_png,
    annotate_ttests=False,
    ttest_pairs=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(4, 4))
    xs = np.arange(len(labels))

    label_to_color = {lab: PALETTE_MEANS[i % len(PALETTE_MEANS)] for i, lab in enumerate(labels)}
    rng = np.random.default_rng(0)

    # Track group summaries for annotation placement
    group_mus = {}
    group_sds = {}
    group_max = {}

    for x, lab in zip(xs, labels):
        # filter rows for this group
        mask = np.ones(len(df_sub), dtype=bool)
        for col, val in zip(group_cols, lab):
            mask &= (df_sub[col] == val)
        cur = df_sub.loc[mask, ["subject_row", "hemi", value_col]].dropna(subset=[value_col])

        if len(cur) == 0:
            continue

        vals  = cur[value_col].to_numpy(dtype=float)
        hemis = cur["hemi"].astype(str).tolist()
        jit   = rng.normal(loc=x, scale=JITTER_STD, size=len(vals))

        # ---- individuals: grey, slightly opaque
        for j in range(len(vals)):
            marker = HEMI_MARK.get(hemis[j], "o")
            if marker == "o":
                ax.scatter(
                    jit[j], vals[j],
                    s=30,
                    alpha=0.35,
                    color="#7f7f7f",
                    marker=marker,
                    linewidths=0,
                    zorder=1,
                )
            else:  # 'x' (Right)
                ax.scatter(
                    jit[j], vals[j],
                    s=30,
                    alpha=0.35,
                    color="#7f7f7f",
                    marker=marker,
                    zorder=1,
                )

        # ---- mean ± SD across both hemispheres in the group
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        c = label_to_color[lab]

        ax.errorbar(
            x=[x], y=[mu],
            yerr=[[sd], [sd]],
            fmt="o",
            markersize=10,
            markerfacecolor=c,
            markeredgecolor="black",
            markeredgewidth=0.9,
            ecolor=c,
            elinewidth=2.8,
            capsize=5,
            capthick=2.8,
            linestyle="none",
            zorder=5,
        )

        group_mus[lab] = mu
        group_sds[lab] = sd
        group_max[lab] = float(np.nanmax(vals))

    # axis/legend
    ax.set_xticks(xs)
    ax.set_xticklabels(["/".join(map(str, lab)) for lab in labels], rotation=0)
    ax.grid(alpha=0.2, axis="y")
    ax.set_title(title, fontsize=10)

    legend_elems = [
        Line2D([0],[0], marker='o', color='w', label='Left (L)',
               markerfacecolor='#7f7f7f', alpha=0.35, markersize=7, linestyle='None'),
        Line2D([0],[0], marker='x', color='#7f7f7f', label='Right (R)',
               alpha=0.35, markersize=7, linestyle='None'),
    ]
    ax.legend(handles=legend_elems, title="Hemisphere", loc="best", frameon=True)

    # ---- Optional: paired t-test annotations (only sensible when x-axis is CONDITIONS)
    if annotate_ttests and ttest_pairs:
        # map condition name -> x position
        cond_to_x = {}
        for x, lab in zip(xs, labels):
            # labels are like (cond,) for these plots
            if len(lab) == 1:
                cond_to_x[str(lab[0])] = x

        y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        base_y = max(group_mus.get(lab, -np.inf) + group_sds.get(lab, 0.0) for lab in labels) if labels else ax.get_ylim()[1]
        if not np.isfinite(base_y):
            base_y = ax.get_ylim()[1]

        step = 0.06 * y_span  # spacing between stacked brackets
        h = 0.015 * y_span

        used = 0
        for (a, b) in ttest_pairs:
            if a not in cond_to_x or b not in cond_to_x:
                continue

            p = paired_ttest_by_subject_mean(df_sub, value_col=value_col, cond_a=a, cond_b=b)
            stars = p_to_stars(p)
            if stars is None:
                continue

            x1, x2 = cond_to_x[a], cond_to_x[b]
            if x2 < x1:
                x1, x2 = x2, x1

            y = base_y + (used + 1) * step
            add_sig_bracket(ax, x1, x2, y, stars, h=h)
            used += 1

        # ensure room for annotations
        if used > 0:
            ax.set_ylim(ax.get_ylim()[0], base_y + (used + 2) * step)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

# ---------------------------
# 1) Consistency: separate plots for MICs and PNI
#    groups = (condition) within each dataset
# ---------------------------
df_cons = df[df["measure"] == "consistency"].copy()
if not df_cons.empty:
    for ds in DATASETS:
        df_ds = df_cons[df_cons["dataset"] == ds].copy()
        labels_c = [(cond,) for cond in CONDITIONS]

        scatter_block(
            df_sub=df_ds.rename(columns={"value": "metric"}),
            value_col="metric",
            labels=labels_c,
            group_cols=["condition"],
            title=f"Consistency (within-subject, across sessions) — {ds}",
            out_png=os.path.join(OUTDIR, f"CGI-consistency_scatter_{ds}.png"),
            annotate_ttests=True,
            ttest_pairs=TTEST_COMPARISONS,
        )

# ---------------------------
# 2) Identifiability: separate plots for MICs and PNI
#    groups = (condition) within each dataset
# ---------------------------
df_ident = df[df["measure"] == "identifiability"].copy()
if not df_ident.empty:
    for ds in DATASETS:
        df_ds = df_ident[df_ident["dataset"] == ds].copy()
        labels_i = [(cond,) for cond in CONDITIONS]

        scatter_block(
            df_sub=df_ds.rename(columns={"value": "metric"}),
            value_col="metric",
            labels=labels_i,
            group_cols=["condition"],
            title=f"Identifiability (between-subject / consistency-normalized) — {ds}",
            out_png=os.path.join(OUTDIR, f"CGI-identifiability_scatter_{ds}.png"),
            annotate_ttests=True,
            ttest_pairs=TTEST_COMPARISONS,
        )

# ---------------------------
# 3) Generalizability: unchanged structure (no single dataset to split by),
#    but mean colors now use only the 3rd/4th/5th palette colors (cycled).
# ---------------------------
df_gen = df[df["measure"] == "generalizability"].copy()
if not df_gen.empty:
    if "dataset_pair" not in df_gen.columns:
        raise RuntimeError("metrics_per_subject.csv lacks 'dataset_pair' for generalizability. Re-run the analysis script.")

    PAIRS = ["PNI-MICs", "MICs-bMICs", "bMICs-PNI"]
    labels_g = [(cond, pair) for cond in CONDITIONS for pair in PAIRS]

    scatter_block(
        df_sub=df_gen.rename(columns={"value": "metric"}),
        value_col="metric",
        labels=labels_g,
        group_cols=["condition", "dataset_pair"],
        title="Generalizability (pairwise correlations per person)",
        out_png=os.path.join(OUTDIR, "CGI-generalizability_scatter.png"),
        annotate_ttests=False,
    )

print(f"Done. Wrote plots to: {OUTDIR}/")