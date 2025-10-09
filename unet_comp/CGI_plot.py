#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ---------------------------
# Config
# ---------------------------
DATASETS   = ["PNI", "MICs", "bMICs"]
CONDITIONS = ["T1w", "synthseg_v0.2"]
JITTER_STD = 0.035
MEAN_S     = 140
DPI        = 160
OUTDIR     = "plots/"
HEMI_MARK  = {"L": "o", "R": "x"}  # hemisphere -> marker

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
# ---------------------------
all_subject_rows = np.unique(df["subject_row"].values)
all_subject_rows = all_subject_rows[~pd.isna(all_subject_rows)]
all_subject_rows = all_subject_rows.astype(int) if len(all_subject_rows) else np.array([], dtype=int)

cmap = plt.get_cmap(args.palette, max(len(all_subject_rows), 1))
color_lookup = {sr: cmap(i % cmap.N) for i, sr in enumerate(sorted(all_subject_rows))}

# ---------------------------
# Jittered scatter helper
# - groups defined by labels & group_cols
# - points colored by subject_row and shaped by hemisphere
# - group mean/SD across BOTH hemispheres (hemi not in group_cols)
# ---------------------------
def scatter_block(df_sub, value_col, labels, group_cols, title, out_png):
    plt.figure(figsize=(8, 4))
    xs = np.arange(len(labels))

    for x, lab in zip(xs, labels):
        # filter rows for this group
        mask = np.ones(len(df_sub), dtype=bool)
        for col, val in zip(group_cols, lab):
            mask &= (df_sub[col] == val)
        cur = df_sub.loc[mask, ["subject_row", "hemi", value_col]].dropna(subset=[value_col])

        if len(cur) > 0:
            vals  = cur[value_col].to_numpy(dtype=float)
            srows = cur["subject_row"].to_numpy(dtype=float)  # may have NaNs; will fall back to gray
            hemis = cur["hemi"].astype(str).tolist()
            jit   = np.random.normal(loc=x, scale=JITTER_STD, size=len(vals))

            # draw each point
            for j in range(len(vals)):
                hemi = hemis[j]
                marker = HEMI_MARK.get(hemi, "o")
                color = color_lookup.get(int(srows[j]) if not np.isnan(srows[j]) else None, "gray")
                # Use edge/face handling so both 'o' and 'x' look good
                if marker == "o":
                    plt.scatter(jit[j], vals[j], s=30, alpha=0.85, color=color, marker=marker, linewidths=0)
                else:  # 'x' (Right)
                    plt.scatter(jit[j], vals[j], s=30, alpha=0.85, color=color, marker=marker)

            # mean & SD across both hemispheres in the group
            mu = np.nanmean(vals)
            sd = np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0
            plt.scatter([x], [mu], s=MEAN_S, color="black", zorder=5)
            plt.errorbar([x], [mu], yerr=[[sd],[sd]], fmt="none",
                         ecolor="black", elinewidth=2, capsize=6, capthick=2)

    # axis/legend
    plt.xticks(xs, ["/".join(map(str, lab)) for lab in labels], rotation=0)
    plt.grid(alpha=0.2, axis="y")

    # hemisphere marker legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', label='Left (L)', markerfacecolor='black', markersize=7),
        Line2D([0],[0], marker='x', color='black', label='Right (R)', markersize=7, linestyle='None')
    ]
    plt.legend(handles=legend_elems, title="Hemisphere", loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI)
    plt.close()

# ---------------------------
# 1) Consistency: groups = (condition, dataset)
# ---------------------------
df_cons = df[df["measure"] == "consistency"].copy()
if not df_cons.empty:
    labels_c = [(cond, ds) for cond in CONDITIONS for ds in DATASETS]
    scatter_block(
        df_sub=df_cons.rename(columns={"value": "metric"}),
        value_col="metric",
        labels=labels_c,
        group_cols=["condition", "dataset"],
        title="Consistency (within-subject, across sessions)",
        out_png=os.path.join(OUTDIR, "CGI-consistency_scatter.png"),
    )

# ---------------------------
# 2) Identifiability: groups = (condition, dataset)
# ---------------------------
df_ident = df[df["measure"] == "identifiability"].copy()
if not df_ident.empty:
    labels_i = [(cond, ds) for cond in CONDITIONS for ds in DATASETS]
    scatter_block(
        df_sub=df_ident.rename(columns={"value": "metric"}),
        value_col="metric",
        labels=labels_i,
        group_cols=["condition", "dataset"],
        title="Identifiability (between-subject / consistency-normalized)",
        out_png=os.path.join(OUTDIR, "CGI-identifiability_scatter.png"),
    )

# ---------------------------
# 3) Generalizability: groups = (condition, dataset_pair)
# ---------------------------
df_gen = df[df["measure"] == "generalizability"].copy()
if not df_gen.empty:
    # Ensure the expected column exists (new CSV has dataset_pair)
    if "dataset_pair" not in df_gen.columns:
        raise RuntimeError("metrics_per_subject.csv lacks 'dataset_pair' for generalizability. Re-run the analysis script.")
    # Fixed order of pairs for x-axis:
    PAIRS = ["PNI-MICs", "MICs-bMICs", "bMICs-PNI"]
    labels_g = [(cond, pair) for cond in CONDITIONS for pair in PAIRS]

    # Reuse scatter_block but with group_cols = ["condition","dataset_pair"]
    scatter_block(
        df_sub=df_gen.rename(columns={"value": "metric"}),
        value_col="metric",
        labels=labels_g,
        group_cols=["condition", "dataset_pair"],
        title="Generalizability (pairwise correlations per person)",
        out_png=os.path.join(OUTDIR, "CGI-generalizability_scatter.png"),
    )

# ---------------------------
# Optional: subject_row -> color legend image
# ---------------------------
if len(all_subject_rows) > 0:
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.12 * len(all_subject_rows)))
    ax.axis("off")
    y = 0.9
    for i, sr in enumerate(sorted(all_subject_rows)):
        ax.scatter(0.02, y, s=60, color=color_lookup[sr])
        ax.text(0.05, y, f"subject_row {sr}", va="center", fontsize=9)
        y -= 1 / (len(all_subject_rows) + 2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "CGI-subject_row_color_legend.png"), dpi=DPI)
    plt.close(fig)

print(f"Done. Wrote plots to: {OUTDIR}/")
