#!/usr/bin/env python3
"""
Canonical folded/unfolded L/R plots per measure using hippomaps.plotting.surfplot_canonical_foldunfold.

Order: thickness -> curvature -> gyrification
Each measure is plotted with ONE global color_range (lo, hi) shared across all datasets/conditions.

NEW:
- Also writes a DIFFERENCE map for each non-T1w condition: (COND - T1w), per dataset/measure
- For synthlayer data: apply special reindexing (reshape to 2:1 rectangle, transpose, flatten)

Scans directories like:
  hippunfold_{DATASET}_{CONDITION}*/sub-*/ses-*/metric/sub-*_ses-*_hemi-{L|R}_den-{DEN}_label-hipp_{measure}.shape.gii
"""

import glob
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from hippomaps.plotting import surfplot_canonical_foldunfold  # key helper

# --- add imports somewhere above (near other imports) ---
import re
import pandas as pd
from statsmodels.stats.anova import AnovaRM

# ---- Config ----
DATASETS_DEFAULT   = ["PNI"]  # , "MICs", "bMICs"]
CONDITIONS_DEFAULT = ["T1w", "synthseg_v0.2", "synthlayer_v0.3"]
HEMIS              = ["L", "R"]
MEASURE_ORDER      = ["thickness", "curv", "gyrification"]
DENSITY_DEFAULT    = "8k"

# Per-measure colormap and pretty label (absolute maps)
CMAP_SINGLE = {
    "thickness": "viridis",
    "gyrification": "plasma",
    "curv": "coolwarm",
}
# Difference-map colormap (COND - T1w)
CMAP_DIFF = "coolwarm"

PRETTY = {"curv": "curvature"}


# -----------------------
# synthlayer reindexing
# -----------------------
def _synthlayer_reindex_2to1_transpose_flatten(arr: np.ndarray) -> np.ndarray:
    """
    Special operation for synthlayer arrays:
      reshape to a 2:1 rectangle, then transpose, then flatten again.

    We infer H and W from V such that W = 2*H and H*W = V.
    Commonly for 8k: V=8192 -> H=64, W=128.

    If we cannot infer a valid 2:1 rectangle, returns arr unchanged (with a warning).
    """
    a = np.asarray(arr, dtype=float).ravel()
    V = a.size
    if V % 2 != 0:
        print(f"[WARN] synthlayer reshape skipped (V not even): V={V}")
        return a

    # Primary inference: V = 2*H^2 -> H = sqrt(V/2)
    H = int(round(np.sqrt(V / 2.0)))
    W = 2 * H
    if H <= 0 or H * W != V:
        # fallback: search for any H that yields W=2H and H*W=V
        found = False
        for h in range(1, int(np.sqrt(V)) + 2):
            w = 2 * h
            if h * w == V:
                H, W = h, w
                found = True
                break
        if not found:
            print(f"[WARN] synthlayer reshape skipped (could not infer 2:1 grid): V={V}")
            return a

    grid = a.reshape(W, H)
    grid = grid.T  # transpose
    return grid.reshape(-1)


# ---- IO helpers ----
def load_gifti_data(shape_path: Path) -> np.ndarray:
    g = nib.load(str(shape_path))
    return np.asarray(g.darrays[0].data, dtype=float)


def load_metric(shape_path: Path, condition: str) -> np.ndarray:
    """
    Load metric vector; apply synthlayer special reindexing if needed.
    """
    v = load_gifti_data(shape_path)
    if "synthlayer" in str(condition):
        v = _synthlayer_reindex_2to1_transpose_flatten(v)
    return v


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


def average_metric(paths, condition: str):
    """Vertexwise mean across subjects (ignoring NaNs). Applies synthlayer reindexing if needed."""
    X = []
    for p in paths:
        try:
            X.append(load_metric(p, condition))
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
                            vals.append(load_metric(fp, cond))
                        except Exception as e:
                            print(f"[WARN] range scan failed {fp}: {e}")
        if vals:
            v = np.concatenate(vals)
            lo, hi = robust_min_max(v, q_lo, q_hi)
            if m in ("curv", "curvature"):
                absmax = max(abs(lo), abs(hi))
                lo, hi = -absmax, absmax
            ranges[m] = (lo, hi)
        else:
            ranges[m] = (0.0, 1.0)
        print(f"[range abs] {m}: {ranges[m][0]:.4g} .. {ranges[m][1]:.4g}")
    return ranges


def compute_global_diff_ranges(root, datasets, conditions, hemis, measures, den, q_lo=2, q_hi=98):
    """
    Global color ranges per measure for DIFF maps: (COND - T1w).
    Uses dataset-level mean maps (not subject-wise paired diffs).
    Symmetric about 0 for all measures (diffs are centered).
    """
    diff_ranges = {}

    non_t1 = [c for c in conditions if c != "T1w"]
    for m in measures:
        diffs = []
        for ds in datasets:
            # Precompute baseline mean maps per hemi for T1w
            base = {}
            for h in hemis:
                fps0 = collect_metric_files(root, ds, "T1w", h, den, m)
                base[h] = average_metric(fps0, "T1w") if fps0 else None

            if all(v is None for v in base.values()):
                continue

            for cond in non_t1:
                for h in hemis:
                    fps = collect_metric_files(root, ds, cond, h, den, m)
                    if not fps:
                        continue
                    cur = average_metric(fps, cond)
                    if cur is None or base[h] is None:
                        continue
                    if len(cur) != len(base[h]):
                        print(f"[WARN] diff range mismatch {ds}/{cond}/{m}/{h}: {len(cur)} vs {len(base[h])}")
                        continue
                    diffs.append(cur - base[h])

        if diffs:
            v = np.concatenate([d.ravel() for d in diffs])
            v = v[np.isfinite(v)]
            if v.size:
                lo, hi = robust_min_max(v, q_lo, q_hi)
                absmax = max(abs(lo), abs(hi))
                diff_ranges[m] = (-absmax, absmax)
            else:
                diff_ranges[m] = (-1.0, 1.0)
        else:
            diff_ranges[m] = (-1.0, 1.0)

        print(f"[range diff] {m}: {diff_ranges[m][0]:.4g} .. {diff_ranges[m][1]:.4g}")

    return diff_ranges


# ---- Main (script-style globals to match your snippet) ----
root = Path(".")
outdir = Path("plots")
outdir.mkdir(parents=True, exist_ok=True)

# Compute global ranges ONCE for all ABS maps
ranges_abs = compute_global_ranges(
    root,
    DATASETS_DEFAULT,
    CONDITIONS_DEFAULT,
    HEMIS,
    MEASURE_ORDER,
    DENSITY_DEFAULT,
    q_lo=2.0,
    q_hi=98.0,
)

# Compute global ranges ONCE for all DIFF maps (COND - T1w)
ranges_diff = compute_global_diff_ranges(
    root,
    DATASETS_DEFAULT,
    CONDITIONS_DEFAULT,
    HEMIS,
    MEASURE_ORDER,
    DENSITY_DEFAULT,
    q_lo=2.0,
    q_hi=98.0,
)

# Plot measure-by-measure
for measure in MEASURE_ORDER:
    pretty = PRETTY.get(measure, measure)
    cmap_abs = CMAP_SINGLE[measure]
    crange_abs = ranges_abs[measure]

    crange_diff = ranges_diff[measure]
    cmap_diff = CMAP_DIFF

    print(f"\n=== Rendering all {pretty} ABS maps with shared color_range={crange_abs} ===")
    print(f"=== Rendering all {pretty} DIFF maps (COND - T1w) with shared color_range={crange_diff} ===")

    for ds in DATASETS_DEFAULT:
        # Precompute T1w baseline mean maps once per dataset/measure
        baseline = {h: None for h in HEMIS}
        for h in HEMIS:
            files0 = collect_metric_files(root, ds, "T1w", h, DENSITY_DEFAULT, measure)
            if files0:
                baseline[h] = average_metric(files0, "T1w")

        for cond in CONDITIONS_DEFAULT:
            # Gather hemisphere means for THIS measure only
            hemi_means = {h: None for h in HEMIS}
            for h in HEMIS:
                files = collect_metric_files(root, ds, cond, h, DENSITY_DEFAULT, measure)
                if files:
                    hemi_means[h] = average_metric(files, cond)

            if all(v is None for v in hemi_means.values()):
                print(f"[INFO] No {pretty} data for {ds}/{cond}; skipping ABS+DIFF.")
                continue

            # ---------- ABS plot ----------
            first = next((arr for arr in hemi_means.values() if arr is not None), None)
            V = len(first)

            cdata = np.full((V, 2, 1), np.nan, dtype=float)
            for hi, h in enumerate(HEMIS):
                arr = hemi_means[h]
                if arr is not None:
                    if len(arr) != V:
                        raise ValueError(f"Vertex count mismatch for {ds}/{cond} {measure} {h}: {len(arr)} vs {V}")
                    cdata[:, hi, 0] = arr

            outfile_abs = outdir / f"hippomaps_{measure}_{ds}_{cond}_den-{DENSITY_DEFAULT}_canonical_folded-unfolded.png"

            surfplot_canonical_foldunfold(
                cdata,
                hemis=HEMIS,
                labels=["hipp"],
                den=DENSITY_DEFAULT,
                color_bar="right",
                color_range=crange_abs,
                share="both",
                cmap=cmap_abs,
                label_text={"right": [pretty]},
                screenshot=True,
                filename=str(outfile_abs),
            )
            print(f"[OK] wrote {outfile_abs}")

            # ---------- DIFF plot (COND - T1w) ----------
            # Only for non-T1w conditions, and only if we have baseline for at least one hemi.
            if cond != "T1w" and not all(v is None for v in baseline.values()):
                diff_means = {h: None for h in HEMIS}
                for h in HEMIS:
                    if hemi_means[h] is None or baseline[h] is None:
                        continue
                    if len(hemi_means[h]) != len(baseline[h]):
                        print(f"[WARN] DIFF skipped due to vertex mismatch {ds}/{cond}/{pretty}/{h}")
                        continue
                    diff_means[h] = hemi_means[h] - baseline[h]

                if all(v is None for v in diff_means.values()):
                    print(f"[INFO] No DIFF data for {ds}/{cond}/{pretty}; skipping DIFF plot.")
                    continue

                first_d = next((arr for arr in diff_means.values() if arr is not None), None)
                Vd = len(first_d)

                cdata_d = np.full((Vd, 2, 1), np.nan, dtype=float)
                for hi, h in enumerate(HEMIS):
                    arr = diff_means[h]
                    if arr is not None:
                        if len(arr) != Vd:
                            raise ValueError(f"Vertex count mismatch (DIFF) for {ds}/{cond} {measure} {h}: {len(arr)} vs {Vd}")
                        cdata_d[:, hi, 0] = arr

                outfile_diff = outdir / f"hippomaps_{measure}_{ds}_{cond}-minus-T1w_den-{DENSITY_DEFAULT}_canonical_folded-unfolded.png"

                surfplot_canonical_foldunfold(
                    cdata_d,
                    hemis=HEMIS,
                    labels=["hipp"],
                    den=DENSITY_DEFAULT,
                    color_bar="right",
                    color_range=crange_diff,
                    share="both",
                    cmap=cmap_diff,
                    label_text={"right": [f"{pretty} (Δ vs T1w)"]},
                    screenshot=True,
                    filename=str(outfile_diff),
                )
                print(f"[OK] wrote {outfile_diff}")


# --- helper: collect all vertex values for a dataset+condition+measure (both hemis) ---
def collect_metric_values(root: Path, dataset: str, condition: str, hemis, den: str, measure: str):
    vals = []
    for h in hemis:
        for fp in collect_metric_files(root, dataset, condition, h, den, measure):
            try:
                v = load_metric(fp, condition)
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
# per-dataset histograms
# =========================
print("\n=== Building per-dataset histograms (shared ranges per measure) ===")
outdir.mkdir(parents=True, exist_ok=True)

NBINS = 60
bins_per_measure = {}
for m in MEASURE_ORDER:
    lo, hi = ranges_abs[m]
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0
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


# --- helper: parse subject + session from a filepath like .../sub-XXX/ses-YYY/... ---
_sub_re = re.compile(r"/sub-([^/]+)/ses-([^/]+)/")
def parse_sub_ses(path: Path):
    m = _sub_re.search(str(path))
    if not m:
        m2 = re.search(r"sub-([A-Za-z0-9]+)[_/]", str(path))
        sub = m2.group(1) if m2 else None
        return sub, None
    return m.group(1), m.group(2)


# --- build a tidy table of subject-level means (per dataset / condition / measure) ---
def build_subject_table(root: Path, datasets, conditions, hemis, measures, den: str) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for m in measures:
            subj_cond_vals = {}
            for cond in conditions:
                for h in hemis:
                    fps = collect_metric_files(root, ds, cond, h, den, m)
                    for fp in fps:
                        sub, ses = parse_sub_ses(fp)
                        if sub is None:
                            continue
                        try:
                            v = load_metric(fp, cond)
                            v = v[np.isfinite(v)]
                            if v.size == 0:
                                continue
                            hemi_mean = float(np.nanmean(v))
                        except Exception as e:
                            print(f"[WARN] RM table load failed {fp}: {e}")
                            continue
                        subj_cond_vals.setdefault(sub, {}).setdefault(cond, []).append(hemi_mean)

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

            counts = subdf.groupby(["subject", "condition"]).size().unstack(fill_value=0)
            complete_subjects = counts[(counts > 0).all(axis=1)].index
            subdf = subdf[subdf["subject"].isin(complete_subjects)]

            subdf = (
                subdf.groupby(["dataset", "measure", "subject", "condition"], as_index=False)["value"]
                     .mean()
            )

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