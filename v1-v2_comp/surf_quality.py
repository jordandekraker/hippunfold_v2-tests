#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import pyvista as pv
from nibabel.nifti1 import intent_codes
import pandas as pd

# Stats
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf

# ---------------------------------------
# Configuration (edit as needed)
# ---------------------------------------

dataset     = "BIDS_PNI"
LABELS      = ["hipp", "dentate"]   # concatenation order: hipp then dentate
hemi_order  = ["L", "R"]
allowed_den = {"8k", "0p5mm"}
OUTDIR      = Path("plots_cellquality_concat")
OUTDIR.mkdir(parents=True, exist_ok=True)

# subjects
with open(f"{dataset}/participants.txt", "r") as f:
    subjects = sorted(set(f.read().split()))

# sessions (define; used for aggregation, not plotting)
session_pairs = [
    ("ses-01", "ses-02"),
]
ALL_SESSIONS = sorted({s for p in session_pairs for s in p})

# ---- Version roots (prefer your example; fall back to common alternates)
def resolve_version_root(ver_key: str) -> Path:
    candidates = []
    if ver_key == "v1.5.1":
        candidates = [
            Path(f"{dataset}/hippunfold_v1.3.0/hippunfold"),
            Path(f"/data/mica3/{dataset}/derivatives/hippunfold_v1.3.0/hippunfold"),
        ]
    elif ver_key == "v2.0.0":
        candidates = [
            Path(f"{dataset}/hippunfold_v2.0.0"),  # e.g., BIDS_MICs/hippunfold_v2.0.0/...
            Path(f"/data/mica3/{dataset}/derivatives/hippunfold_v2.0.0"),
            Path(f"/export03/data/opt/hippunfold_v2stable/v1-v2_comp/{dataset}/hippunfold_v2.0.0beta"),
        ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else Path(".")

version_dirs = {
    "v1.5.1": resolve_version_root("v1.5.1"),
    "v2.0.0": resolve_version_root("v2.0.0"),
}
print("[INFO] version roots:")
for k, p in version_dirs.items():
    print(f"  {k}: {p}")

# plotting style
JITTER_STD = 0.035
MEAN_S     = 140
PALETTE    = "jet"  # per-subject color map
HEMI_MARK  = {"L": "o", "R": "x"}
RNG        = np.random.default_rng(42)

# ---------------------------------------
# IO: GIFTI -> verts/faces
# ---------------------------------------

def load_gifti_verts_faces(filepath: Path):
    gii = nib.load(str(filepath))
    verts, faces = None, None
    for arr in gii.darrays:
        if arr.intent == intent_codes['NIFTI_INTENT_POINTSET']:
            verts = np.asarray(arr.data, dtype=np.float64)
        elif arr.intent == intent_codes['NIFTI_INTENT_TRIANGLE']:
            faces = np.asarray(arr.data, dtype=np.int64)
    if verts is None or faces is None:
        raise ValueError(f"Missing POINTSET or TRIANGLE in {filepath}")
    return verts, faces

def find_surface(root_dir: Path, subj: str, ses: str, hemi: str, label: str):
    """Prefer 8k; accept 0p5mm; otherwise any allowed den."""
    surf_dir = root_dir / f"sub-{subj}" / f"{ses}" / "surf"
    patterns = [
        f"sub-{subj}_{ses}_hemi-{hemi}_space-T1w_den-8k_label-{label}_midthickness.surf.gii",
        f"sub-{subj}_{ses}_hemi-{hemi}_space-T1w_den-0p5mm_label-{label}_midthickness.surf.gii",
    ]
    for pat in patterns:
        m = list(surf_dir.glob(pat))
        if m:
            return m[0]
    glob_pat = surf_dir / f"sub-{subj}_{ses}_hemi-{hemi}_space-T1w_den-*_label-{label}_midthickness.surf.gii"
    matches = sorted(glob_pat.parent.glob(glob_pat.name))
    matches = [m for m in matches if any(f"den-{d}" in m.name for d in allowed_den)]
    return matches[0] if matches else None

# ---------------------------------------
# Concatenate hipp + dentate -> PolyData -> cell quality (default)
# ---------------------------------------

def concat_two_meshes_to_polydata(path_a: Path, path_b: Path) -> pv.PolyData:
    """Concatenate two triangle meshes (A then B) into a single PolyData."""
    Va, Fa = load_gifti_verts_faces(path_a)
    Vb, Fb = load_gifti_verts_faces(path_b)
    V = np.vstack([Va, Vb])
    Fb_off = Fb + Va.shape[0]
    F = np.vstack([Fa, Fb_off])
    faces_vtk = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
    return pv.PolyData(V, faces_vtk)

records = []  # rows: version, hemi, subject_id, session, quality_mean, quality_std

for ver, ver_root in version_dirs.items():
    if not ver_root.exists():
        print(f"[WARN] Version root not found for {ver}: {ver_root}")
    for hemi in hemi_order:
        for subj in subjects:
            for ses in ALL_SESSIONS:
                surf_hipp    = find_surface(ver_root, subj, ses, hemi, "hipp")
                surf_dentate = find_surface(ver_root, subj, ses, hemi, "dentate")
                if surf_hipp is None or surf_dentate is None:
                    continue
                try:
                    poly = concat_two_meshes_to_polydata(surf_hipp, surf_dentate)
                    qmesh = poly.compute_cell_quality()  # default quality measure
                    q = np.asarray(qmesh["CellQuality"], dtype=float)
                    if q.size == 0:
                        continue
                    mu = float(np.mean(q))
                    sd = float(np.std(q, ddof=1)) if q.size > 1 else 0.0
                    records.append([ver, hemi, subj, ses, mu, sd])
                except Exception as e:
                    print(f"[WARN] Failed on {surf_hipp} + {surf_dentate}: {e}")

if not records:
    print("[WARN] No concatenated mesh quality records found. Check paths and file patterns.")

df = pd.DataFrame(records, columns=[
    "version","hemi","subject_id","session","quality_mean","quality_std"
])

# ---------------------------------------
# Plot: jittered scatter by version (aggregate across sessions)
# ---------------------------------------

# subject colors
all_subjects = sorted(df["subject_id"].dropna().unique()) if not df.empty else []
cmap = plt.get_cmap(PALETTE, max(len(all_subjects), 1))
color_lookup = {sid: cmap(i % cmap.N) for i, sid in enumerate(all_subjects)}

def scatter_block(df_sub, value_col, labels, group_cols, title, out_png):
    plt.figure(figsize=(9.5, 6))
    xs = np.arange(len(labels))
    for x, lab in zip(xs, labels):
        mask = np.ones(len(df_sub), dtype=bool)
        for col, val in zip(group_cols, lab):
            mask &= (df_sub[col] == val)
        cur = df_sub.loc[mask, ["subject_id","hemi",value_col]].dropna(subset=[value_col])
        if len(cur) > 0:
            vals  = cur[value_col].astype(float).to_numpy()
            subs  = cur["subject_id"].astype(str).to_numpy()
            hemis = cur["hemi"].astype(str).tolist()
            jit   = RNG.normal(loc=x, scale=JITTER_STD, size=len(vals))
            for j in range(len(vals)):
                marker = HEMI_MARK.get(hemis[j], "o")
                color  = color_lookup.get(subs[j], "gray")
                if marker == "o":
                    plt.scatter(jit[j], vals[j], s=30, alpha=0.85, color=color, marker=marker, linewidths=0)
                else:
                    plt.scatter(jit[j], vals[j], s=30, alpha=0.85, color=color, marker=marker)
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            plt.scatter([x], [mu], s=MEAN_S, color="black", zorder=5)
            plt.errorbar([x], [mu], yerr=[[sd],[sd]], fmt="none",
                         ecolor="black", elinewidth=2, capsize=6, capthick=2)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', label='Left (L)', markerfacecolor='black', markersize=7),
        Line2D([0],[0], marker='x', color='black', label='Right (R)', markersize=7, linestyle='None')
    ]
    plt.legend(handles=legend_elems, title="Hemisphere", loc="best", frameon=True)
    plt.title(title)
    plt.xticks(xs, ["/".join(map(str, lab)) for lab in labels])
    plt.ylabel(value_col.replace("_"," ").title())
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

if df.empty:
    print("[WARN] Nothing to plot.")
else:
    # Aggregate across sessions → one value per (version, hemi, subject)
    df_plot = (
        df.groupby(["version","hemi","subject_id"], as_index=False)["quality_mean"]
          .mean()
          .rename(columns={"quality_mean":"metric"})
    )
    labels_by_version = [(ver,) for ver in version_dirs.keys()]
    scatter_block(
        df_sub=df_plot,
        value_col="metric",
        labels=labels_by_version,
        group_cols=["version"],
        title=f"Concatenated Mesh Cell Quality (hipp+dentate, VTK default) — {dataset}",
        out_png=str(OUTDIR / f"cellquality_concat_scatter_{dataset}.png"),
    )
    print(f"Saved: {OUTDIR / f'cellquality_concat_scatter_{dataset}.png'}")

# ---------------------------------------
# Repeated-measures ANOVA (within: version; collapse across hemi & session)
# ---------------------------------------

def run_rm_version_only(df_plot_src: pd.DataFrame, subject_col: str = "subject_id"):
    """
    Collapse across hemisphere and session to 1 obs per subject×version,
    then run RM-ANOVA with within-subject factor version.
    Fallback to MixedLM if AnovaRM fails or data are unbalanced.
    """
    if df_plot_src.empty:
        print("\n[RM-ANOVA] No data available.")
        return

    # Start from per-session rows (df): collapse across hemi & session
    df_cells = (
        df.groupby(["subject_id","version"], as_index=False)["quality_mean"]
          .mean()
          .rename(columns={"quality_mean":"cell_mean"})
          .dropna(subset=["cell_mean"])
    )

    # Keep only subjects with all version levels present (balanced RM)
    ver_levels = sorted(df_cells["version"].unique())
    need = len(ver_levels)
    subj_counts = df_cells.groupby(subject_col).size()
    complete_subjects = subj_counts[subj_counts == need].index.tolist()
    df_bal = df_cells[df_cells[subject_col].isin(complete_subjects)].copy()

    print("\nRepeated-measures ANOVA for concatenated cell quality (within: version; hemi/session collapsed)")
    print(f"Subjects total: {df_cells[subject_col].nunique()} | complete for RM: {len(complete_subjects)}")
    print(f"Version levels: {ver_levels}")

    if len(complete_subjects) >= 2:
        try:
            aov = AnovaRM(
                data=df_bal,
                depvar="cell_mean",
                subject=subject_col,
                within=["version"]
            ).fit()
            print(aov.anova_table)
            return
        except Exception as e:
            print(f"[Warning] AnovaRM failed ({e}). Falling back to MixedLM.")

    # Mixed model fallback: random intercept per subject; use all available data
    df_cells["version"] = df_cells["version"].astype("category")
    md = smf.mixedlm("cell_mean ~ C(version)", df_cells, groups=df_cells[subject_col])
    mdf = md.fit(method="lbfgs", reml=True, maxiter=200, disp=False)
    print(mdf.summary())

# Run it
run_rm_version_only(df)
