#!/usr/bin/env python3
"""
Canonical folded/unfolded L/R plots per measure using hippomaps.plotting.surfplot_canonical_foldunfold.

Order: thickness -> curvature -> gyrification
Each measure is plotted with ONE global color_range (lo, hi) shared across all datasets/conditions.

Scans directories like:
  hippunfold_{DATASET}_{CONDITION}*/sub-*/ses-*/metric/sub-*_ses-*_hemi-{L|R}_den-{DEN}_label-hipp_{measure}.shape.gii
"""

import glob
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from hippomaps.plotting import surfplot_canonical_foldunfold  # key helper

# ---- Config ----
DATASETS_DEFAULT   = ["PNI"]#, "MICs", "bMICs"]
CONDITIONS_DEFAULT = ["T1w", "synthseg_v0.2"]
HEMIS              = ["L", "R"]
# We'll iterate in this order to satisfy "thickness first, then curvature, etc."
MEASURE_ORDER      = ["thickness", "curv", "gyrification"]
DENSITY_DEFAULT    = "8k"

# Per-measure colormap and pretty label
CMAP_SINGLE = {
    "thickness": "viridis",
    "gyrification": "plasma",
    "curv": "coolwarm",
}
PRETTY = {"curv": "curvature"}

# ---- IO helpers ----
def load_gifti_data(shape_path: Path) -> np.ndarray:
    g = nib.load(str(shape_path))
    return np.asarray(g.darrays[0].data, dtype=float)

def robust_min_max(values: np.ndarray, lo=1.0, hi=99.0):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))

def collect_metric_files(root: Path, dataset: str, condition: str, hemi: str, den: str, measure: str):
    """Find .shape.gii for a given combo. Accept both curv and curvature on disk."""
    variants = ["curv", "curvature"] if measure in ("curv", "curvature") else [measure]
    hits = []
    containers = [
        f"hippunfold_{dataset}_{condition}*",
        f"hippunfold_{dataset}_{condition}",
    ]
    for cont in containers:
        for var in variants:
            patt = root / f"{cont}/sub-*/ses-*/metric/sub-*_ses-*_hemi-{hemi}_den-{den}_label-hipp_{var}.shape.gii"
            hits.extend([Path(p) for p in glob.glob(str(patt))])
    return sorted(set(hits))

def average_metric(paths):
    """Vertexwise mean across subjects (ignoring NaNs)."""
    X = []
    for p in paths:
        try:
            X.append(load_gifti_data(p))
        except Exception as e:
            print(f"[WARN] failed to load {p}: {e}")
    if not X:
        return None
    X = np.vstack([x[np.newaxis, :] for x in X])  # S × V
    with np.errstate(invalid="ignore"):
        return np.nanmean(X, axis=0)

def compute_global_ranges(root, datasets, conditions, hemis, measures, den, q_lo=2, q_hi=98):
    """Global color ranges per measure (curvature symmetric). Returns dict[measure] -> (lo, hi)."""
    ranges = {}
    for m in measures:
        vals = []
        for ds in datasets:
            for cond in conditions:
                for h in hemis:
                    for fp in collect_metric_files(root, ds, cond, h, den, m):
                        try:
                            vals.append(load_gifti_data(fp))
                        except Exception as e:
                            print(f"[WARN] range scan failed {fp}: {e}")
        if vals:
            v = np.concatenate(vals)
            lo, hi = robust_min_max(v, q_lo, q_hi)
            if m in ("curv", "curvature"):
                # symmetric about 0 for curvature
                absmax = max(abs(lo), abs(hi))
                lo, hi = -absmax, absmax
            ranges[m] = (lo, hi)
        else:
            ranges[m] = (0.0, 1.0)
        print(f"[range] {m}: {ranges[m][0]:.4g} .. {ranges[m][1]:.4g}")
    return ranges

# ---- Main (script-style globals to match your snippet) ----
root = Path(".")
outdir = Path("plots")
outdir.mkdir(parents=True, exist_ok=True)

# Compute global ranges ONCE for all data (thickness same everywhere, etc.)
ranges = compute_global_ranges(
    root,
    DATASETS_DEFAULT,
    CONDITIONS_DEFAULT,
    HEMIS,
    MEASURE_ORDER,
    DENSITY_DEFAULT,
    q_lo=2.0,
    q_hi=98.0,
)

# Plot measure-by-measure to use a single (lo, hi) for all its panels
for measure in MEASURE_ORDER:
    pretty = PRETTY.get(measure, measure)
    cmap = CMAP_SINGLE[measure]
    crange = ranges[measure]  # single (lo, hi) tuple

    print(f"\n=== Rendering all {pretty} maps with shared color_range={crange} ===")
    for ds in DATASETS_DEFAULT:
        for cond in CONDITIONS_DEFAULT:
            # Gather hemisphere means for THIS measure only
            hemi_means = {h: None for h in HEMIS}
            for h in HEMIS:
                files = collect_metric_files(root, ds, cond, h, DENSITY_DEFAULT, measure)
                if files:
                    hemi_means[h] = average_metric(files)

            if all(v is None for v in hemi_means.values()):
                print(f"[INFO] No {pretty} data for {ds}/{cond}; skipping.")
                continue

            # Determine vertex count V from whichever hemi is present
            first = next((arr for arr in hemi_means.values() if arr is not None), None)
            V = len(first)

            # Assemble single-feature cdata: V × 2 × 1
            cdata = np.full((V, 2, 1), np.nan, dtype=float)
            for hi, h in enumerate(HEMIS):
                arr = hemi_means[h]
                if arr is not None:
                    if len(arr) != V:
                        raise ValueError(f"Vertex count mismatch for {ds}/{cond} {measure} {h}: {len(arr)} vs {V}")
                    cdata[:, hi, 0] = arr

            outfile = outdir / f"hippomaps_{measure}_{ds}_{cond}_den-{DENSITY_DEFAULT}_canonical_folded-unfolded.png"

            # One call per measure with a SINGLE (lo, hi) color_range
            # filename=... lets hippomaps/brainspace handle saving
            surfplot_canonical_foldunfold(
                cdata,
                hemis=HEMIS,
                labels=["hipp"],
                den=DENSITY_DEFAULT,
                color_bar="right",
                color_range=crange,              # <-- single (lo, hi)
                share="both",
                cmap=cmap,                       # single colormap string
                label_text={"right": [pretty]},
                screenshot=True,
                filename=str(outfile),
            )
            print(f"[OK] wrote {outfile}")


# --- helper: collect all vertex values for a dataset+condition+measure (both hemis) ---
def collect_metric_values(root: Path, dataset: str, condition: str, hemis, den: str, measure: str):
    vals = []
    for h in hemis:
        for fp in collect_metric_files(root, dataset, condition, h, den, measure):
            try:
                v = load_gifti_data(fp)
                if v is not None:
                    v = np.asarray(v, dtype=float)
                    v = v[np.isfinite(v)]
                    if v.size:
                        vals.append(v)
            except Exception as e:
                print(f"[WARN] histogram load failed {fp}: {e}")
    if not vals:
        return np.array([], dtype=float)
    return np.concatenate(vals)

# =========================
# NEW: per-dataset histograms
# =========================
print("\n=== Building per-dataset histograms (shared ranges per measure) ===")
outdir.mkdir(parents=True, exist_ok=True)

# choose consistent binning per measure (same across all datasets/conditions)
NBINS = 60
bins_per_measure = {}
for m in MEASURE_ORDER:
    lo, hi = ranges[m]
    # guard against degenerate ranges
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0
    # for curvature ensure symmetry (already enforced in ranges, but re-check)
    if m in ("curv", "curvature"):
        a = max(abs(lo), abs(hi))
        lo, hi = -a, a
    bins_per_measure[m] = np.linspace(lo, hi, NBINS + 1)

for ds in DATASETS_DEFAULT:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for i, m in enumerate(MEASURE_ORDER):
        ax = axes[i]
        m_pretty = PRETTY.get(m, m)
        bins = bins_per_measure[m]

        # pull values for each condition
        plotted_any = False
        for cond in CONDITIONS_DEFAULT:
            v = collect_metric_values(root, ds, cond, HEMIS, DENSITY_DEFAULT, m)
            if v.size == 0:
                print(f"[INFO] No values for {ds}/{cond}/{m_pretty}; skipping that trace.")
                continue
            ax.hist(
                v,
                bins=bins,
                histtype="stepfilled",
                alpha=0.45,
                density=True,
                label=cond,
            )
            plotted_any = True

        ax.set_title(m_pretty, fontsize=11)
        ax.set_xlabel(m_pretty)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25, linestyle="--")
        if plotted_any:
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    hist_out = outdir / f"hippomaps_hist_{ds}_den-{DENSITY_DEFAULT}.png"
    fig.suptitle(f"Histogram overlays — {ds} (bins shared per measure)", fontsize=12)
    fig.savefig(hist_out, dpi=200)
    plt.close(fig)
    print(f"[OK] wrote {hist_out}")




# --- add imports somewhere above (near other imports) ---
import re
import pandas as pd
from statsmodels.stats.anova import AnovaRM

# --- helper: parse subject + session from a filepath like .../sub-XXX/ses-YYY/... ---
_sub_re = re.compile(r"/sub-([^/]+)/ses-([^/]+)/")
def parse_sub_ses(path: Path):
    m = _sub_re.search(str(path))
    if not m:
        # fallback: try just sub-XXX_... patterns in filename
        m2 = re.search(r"sub-([A-Za-z0-9]+)[_/]", str(path))
        sub = m2.group(1) if m2 else None
        return sub, None
    return m.group(1), m.group(2)

# --- build a tidy table of subject-level means (per dataset / condition / measure) ---
def build_subject_table(root: Path, datasets, conditions, hemis, measures, den: str) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for m in measures:
            # subject-session aggregator per condition to then average across sessions & hemis
            # structure: subj -> cond -> list of hemi means (possibly multiple sessions)
            subj_cond_vals = {}
            for cond in conditions:
                for h in hemis:
                    fps = collect_metric_files(root, ds, cond, h, den, m)
                    for fp in fps:
                        sub, ses = parse_sub_ses(fp)
                        if sub is None:
                            continue
                        try:
                            v = load_gifti_data(fp)
                            v = v[np.isfinite(v)]
                            if v.size == 0:
                                continue
                            hemi_mean = float(np.nanmean(v))
                        except Exception as e:
                            print(f"[WARN] RM table load failed {fp}: {e}")
                            continue
                        subj_cond_vals.setdefault(sub, {}).setdefault(cond, []).append(hemi_mean)

            # collapse hemis & sessions -> one value per (subject, condition)
            for sub, cond_dict in subj_cond_vals.items():
                for cond, vals in cond_dict.items():
                    if len(vals) == 0:
                        continue
                    rows.append({
                        "dataset": ds,
                        "measure": PRETTY.get(m, m),
                        "subject": sub,
                        "condition": cond,
                        "value": float(np.mean(vals)),
                    })
    return pd.DataFrame(rows)

# --- build table and run RM-ANOVA per dataset×measure ---
df_rm = build_subject_table(root, DATASETS_DEFAULT, CONDITIONS_DEFAULT, HEMIS, MEASURE_ORDER, DENSITY_DEFAULT)

summary_rows = []
print("\n=== Repeated-measures ANOVA: value ~ condition (within-subject) ===")
if df_rm.empty:
    print("[INFO] No data available for RM-ANOVA.")
else:
    for ds in DATASETS_DEFAULT:
        for m in [PRETTY.get(x, x) for x in MEASURE_ORDER]:
            subdf = df_rm[(df_rm["dataset"] == ds) & (df_rm["measure"] == m)].copy()
            if subdf.empty:
                print(f"[INFO] Skipping ANOVA for {ds} / {m}: no rows.")
                continue
            # keep only subjects present in ALL conditions (balanced RM)
            counts = subdf.groupby(["subject", "condition"]).size().unstack(fill_value=0)
            complete_subjects = counts[(counts > 0).all(axis=1)].index
            subdf = subdf[subdf["subject"].isin(complete_subjects)]

            # if multiple rows per subject×condition remain (e.g., duplicate sessions), average them
            subdf = (
                subdf.groupby(["dataset", "measure", "subject", "condition"], as_index=False)["value"]
                     .mean()
            )
            # need at least 2 subjects and 2 conditions
            if subdf["subject"].nunique() < 2 or subdf["condition"].nunique() < 2:
                print(f"[INFO] Skipping ANOVA for {ds} / {m}: insufficient balanced subjects.")
                continue

            try:
                res = AnovaRM(
                    data=subdf,
                    depvar="value",
                    subject="subject",
                    within=["condition"],
                ).fit()
                print(f"\n--- {ds} / {m} ---")
                print(res.summary())

                # pull the condition row for a tidy summary
                anova_table = res.anova_table.reset_index().rename(columns={"index": "Effect"})
                cond_row = anova_table[anova_table["Effect"] == "condition"]
                if not cond_row.empty:
                    summary_rows.append({
                        "dataset": ds,
                        "measure": m,
                        "Effect": "condition",
                        "F": float(cond_row["F Value"].values[0]),
                        "Num DF": float(cond_row["Num DF"].values[0]),
                        "Den DF": float(cond_row["Den DF"].values[0]),
                        "Pr > F": float(cond_row["Pr > F"].values[0]),
                        "N subjects": int(subdf["subject"].nunique()),
                    })
            except Exception as e:
                print(f"[WARN] ANOVA failed for {ds} / {m}: {e}")

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).sort_values(["dataset", "measure"])
        csv_out = outdir / "anova_summary.csv"
        df_summary.to_csv(csv_out, index=False)
        print(f"\n[OK] Wrote RM-ANOVA summary: {csv_out}")
    else:
        print("\n[INFO] No ANOVA results to summarize.")
