#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from nibabel.nifti1 import intent_codes
import SimpleITK as sitk
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------------------------------------
# Configuration (edit as needed)
# ---------------------------------------

dataset = "BIDS_MICs"
LABELS = ["hipp", "dentate"]  # concat order: hipp then dentate

# subject list (space/newline-separated IDs)
with open(f"{dataset}/participants.txt", "r") as f:
    subjects = sorted(set(f.read().split()))

# session pairs to define “consistency” observations (use "ses-XX" tokens as-is)
session_pairs = [
    ("ses-01", "ses-02"),
    # ("ses-02", "ses-03"),
    # ("ses-03", "ses-01"),
]

# hippunfold versions
version_dirs = {
    "v1.5.1": Path(f"/data/mica3/{dataset}/derivatives/hippunfold_v1.3.0/hippunfold"),
    "v2.0.0": Path(f"/export03/data/opt/hippunfold_v2-tests/v1-v2_comp/{dataset}/hippunfold_v2.0.0"),
}

hemi_order  = ["L", "R"]
allowed_den = {"8k", "0p5mm"}   # accept either mesh density
OUTDIR = Path("plots_versions")
OUTDIR.mkdir(parents=True, exist_ok=True)

# plotting style
JITTER_STD = 0.035
MEAN_S     = 140
PALETTE    = "jet"  # per-subject color map
HEMI_MARK  = {"L": "o", "R": "x"}
RNG        = np.random.default_rng(42)

# ---------------------------------------
# Affine utilities (ANTs -> MNI, inverted)
# ---------------------------------------

def micapipe_root_for_affine(ds: str) -> str:
    """Use MICs affines for bMICs; otherwise map to dataset itself."""
    if ds == "BIDS_bMICs":
        return "/data/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0"
    return f"/data/mica3/{ds}/derivatives/micapipe_v0.2.0"

def read_ants_rigid_4x4(mat_path: str, invert: bool = True):
    """
    Read an ANTs 0GenericAffine.mat and return a 4x4 *rigid-only* affine (rotation + translation).
    - Uses SVD-based polar decomposition to strip scale/shear.
    - Respects the ITK center. Builds A = [[R, t + c - R c],[0,0,0,1]].
    - If invert=True, inverts the transform first, then extracts rigid from that mapping.
    """
    try:
        T = sitk.ReadTransform(mat_path)
    except Exception:
        return None

    # Coerce to affine
    try:
        T = sitk.AffineTransform(T) if not isinstance(T, sitk.AffineTransform) else T
    except Exception:
        return None
    if T.GetDimension() != 3:
        return None

    # Invert first if you want the rigid that best-approximates the desired direction
    if invert:
        try:
            T = T.GetInverse()
        except Exception:
            return None

    # Extract ITK params
    M = np.array(T.GetMatrix(), dtype=float).reshape(3, 3)
    c = np.array(T.GetCenter(), dtype=float)
    t = np.array(T.GetTranslation(), dtype=float)

    # Rigid: closest rotation to M via SVD (polar decomposition)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Ensure a proper rotation (det +1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    # Build 4x4 with same center+translation convention
    A = np.eye(4, dtype=float)
    A[:3, :3] = R
    A[:3, 3]  = t + c - R @ c
    return A

def apply_affine(points_xyz: np.ndarray, A: np.ndarray) -> np.ndarray:
    N = points_xyz.shape[0]
    homo = np.c_[points_xyz, np.ones((N, 1), dtype=points_xyz.dtype)]
    return (A @ homo.T).T[:, :3]

def affine_for(ds: str, subj: str, ses: str):
    # ses is "ses-XX" already; use as-is
    root = micapipe_root_for_affine(ds)
    mat_path = os.path.join(
        root, f"sub-{subj}", f"{ses}", "xfm",
        f"sub-{subj}_{ses}_from-nativepro_brain_to-MNI152_0.8mm_mode-image_desc-SyN_0GenericAffine.mat",
    )
    return read_ants_rigid_4x4(mat_path, invert=True)

# ---------------------------------------
# IO utilities
# ---------------------------------------

def load_gifti_vertices(filepath: Path):
    gii = nib.load(str(filepath))
    verts = next(arr.data for arr in gii.darrays if arr.intent == intent_codes['NIFTI_INTENT_POINTSET'])
    return np.asarray(verts, dtype=np.float64)  # (N,3)

def find_surface(root_dir: Path, subj: str, ses: str, hemi: str, label: str):
    """
    Find a midthickness surface for a subject/session/hemi/label,
    accepting either den-8k or den-0p5mm (prefers 8k when both exist).
    """
    surf_dir = root_dir / f"sub-{subj}" / f"{ses}" / "surf"
    patterns = [
        f"sub-{subj}_{ses}_hemi-{hemi}_space-T1w_den-8k_label-{label}_midthickness.surf.gii",
        f"sub-{subj}_{ses}_hemi-{hemi}_space-T1w_den-0p5mm_label-{label}_midthickness.surf.gii",
    ]
    for pat in patterns:
        matches = list(surf_dir.glob(pat))
        if matches:
            return matches[0]
    # fallback: any allowed den for this label
    glob_pat = surf_dir / f"sub-{subj}_{ses}_hemi-{hemi}_space-T1w_den-*_label-{label}_midthickness.surf.gii"
    matches = sorted(glob_pat.parent.glob(glob_pat.name))
    matches = [m for m in matches if any(f"den-{d}" in m.name for d in allowed_den)]
    return matches[0] if matches else None


# ---------------------------------------
# Correlation helpers
# ---------------------------------------

def pearsonr_nan(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    a2, b2 = a[m], b[m]
    sa, sb = a2.std(), b2.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a2, b2)[0, 1])

# ---------------------------------------
# Core helpers: session vectors (no averaging), and optional overall-means for matrices
# ---------------------------------------

def get_session_vector(ver_root: Path, ds: str, subj: str, ses: str, hemi: str):
    """
    Load one session’s midthickness for both labels (hipp, dentate), transform to MNI,
    0-center EACH label separately, flip L-x, flatten, and concatenate [hipp, dentate].
    Returns a 1D vector or None if any required piece is missing.
    """
    A = affine_for(ds, subj, ses)
    if A is None:
        return None

    parts = []
    for label in LABELS:
        surf = find_surface(ver_root, subj, ses, hemi, label)
        if surf is None:
            return None
        verts = load_gifti_vertices(surf)
        if verts.ndim != 2 or verts.shape[1] != 3:
            return None

        # Transform to MNI, center per-label, unify hemis by flipping x for L
        verts_mni = apply_affine(verts, A)
        verts_norm = verts_mni - np.mean(verts_mni, axis=0, keepdims=True)
        if hemi == "L":
            verts_norm[:, 0] *= -1
        parts.append(verts_norm.reshape(-1))

    # Concatenate hipp then dentate
    return np.concatenate(parts, axis=0)

def collect_all_sessions(version_dirs, ds, hemi):
    """
    One-pass collection of all available session vectors per subject, per version, per hemisphere.
    No session averaging, no length enforcement here.
    Returns: {version: {subject: [np.ndarray, ...]}}
    """
    all_sessions = sorted({s for pair in session_pairs for s in pair})
    out = {}
    for ver, root in version_dirs.items():
        subj_sess = {}
        for subj in subjects:
            vecs = []
            for ses in all_sessions:
                v = get_session_vector(root, ds, subj, ses, hemi)
                if v is not None:
                    vecs.append(v)
            subj_sess[subj] = vecs  # may be empty
        out[ver] = subj_sess
    return out

def collect_overall_means(version_dirs, ds, hemi):
    """
    For visualization (corr matrices): per-version per-hemi overall mean signature per subject,
    averaged across sessions. This is separate from identifiability, which no longer averages.
    Returns: {version: {subject: np.ndarray or None}}
    """
    all_sessions = sorted({s for pair in session_pairs for s in pair})
    out = {}
    for ver, root in version_dirs.items():
        subj_means = {}
        # determine expected length from first valid subject
        expected_len = None
        for subj in subjects:
            vecs = []
            for ses in all_sessions:
                v = get_session_vector(root, ds, subj, ses, hemi)
                if v is not None:
                    vecs.append(v)
            if vecs:
                Ls = {vv.size for vv in vecs}
                if len(Ls) == 1:
                    expected_len = vecs[0].size
                    break
        for subj in subjects:
            vecs = []
            for ses in all_sessions:
                v = get_session_vector(root, ds, subj, ses, hemi)
                if v is not None:
                    vecs.append(v)
            vecs = [v for v in vecs if expected_len is None or v.size == expected_len]
            if not vecs or expected_len is None:
                subj_means[subj] = None
            else:
                subj_means[subj] = np.nanmean(np.vstack(vecs), axis=0)
        out[ver] = subj_means
    return out

# Build banks
session_bank  = {hemi: collect_all_sessions(version_dirs, dataset, hemi) for hemi in hemi_order}
overall_means = {hemi: collect_overall_means(version_dirs, dataset, hemi) for hemi in hemi_order}

# ---------------------------------------
# Consistency & Identifiability per PAIR (independent observations)
# ---------------------------------------

rows_cons = []   # columns: version, hemi, subject_id, pair, consistency
rows_ident = []  # columns: version, hemi, subject_id, pair, identifiability

for ver, ver_root in version_dirs.items():
    for hemi in hemi_order:
        for subj in subjects:
            for (sesA, sesB) in session_pairs:
                vA = get_session_vector(ver_root, dataset, subj, sesA, hemi)
                vB = get_session_vector(ver_root, dataset, subj, sesB, hemi)
                if vA is None or vB is None or vA.size != vB.size or vA.size < 3:
                    continue

                # Consistency for this pair
                C_pair = pearsonr_nan(vA, vB)
                if np.isnan(C_pair):
                    continue
                rows_cons.append([ver, hemi, subj, f"{sesA}_{sesB}", float(C_pair)])

                # Pair-mean signature for this subject
                m_this_pair = np.nanmean(np.vstack([vA, vB]), axis=0)

                # Between-subject similarity: mean of correlations to ALL sessions of other subjects (same ver/hemi)
                bank = session_bank[hemi][ver]
                cors = []
                for other in subjects:
                    if other == subj:
                        continue
                    for v_other in bank.get(other, []):
                        if v_other is None or v_other.size != m_this_pair.size:
                            continue
                        rbo = pearsonr_nan(m_this_pair, v_other)
                        if not np.isnan(rbo):
                            cors.append(rbo)
                B_pair = np.mean(cors)

                I_pair = ((C_pair - B_pair) / C_pair) if (not np.isnan(B_pair) and C_pair != 0) else np.nan
                rows_ident.append([ver, hemi, subj, f"{sesA}_{sesB}", I_pair])

df_cons = pd.DataFrame(rows_cons,  columns=["version","hemi","subject_id","pair","consistency"])
df_ident= pd.DataFrame(rows_ident, columns=["version","hemi","subject_id","pair","identifiability"])

# ---------------------------------------
# Save CSV
# ---------------------------------------

df_all = pd.concat([
    df_cons.assign(measure="consistency",      value=df_cons["consistency"])[["version","hemi","subject_id","pair","measure","value"]],
    df_ident.assign(measure="identifiability", value=df_ident["identifiability"])[["version","hemi","subject_id","pair","measure","value"]],
], ignore_index=True)

csv_path = f"metrics_per_subject_versions_{dataset}_pairs.csv"
df_all.to_csv(csv_path, index=False)
print(f"Wrote {csv_path}")

# ---------------------------------------
# Plots: jittered scatters (subject color; hemisphere marker), per version
# ---------------------------------------

# color per subject
all_subjects = sorted(df_all["subject_id"].dropna().unique())
cmap = plt.get_cmap(PALETTE, max(len(all_subjects), 1))
color_lookup = {sid: cmap(i % cmap.N) for i, sid in enumerate(all_subjects)}

def scatter_block(df_sub, value_col, labels, group_cols, title, out_png):
    plt.figure(figsize=(3, 4))
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
            # scatter points
            for j in range(len(vals)):
                marker = HEMI_MARK.get(hemis[j], "o")
                color  = color_lookup.get(subs[j], "gray")
                if marker == "o":
                    plt.scatter(jit[j], vals[j], s=30, alpha=0.85, color=color, marker=marker, linewidths=0)
                else:
                    plt.scatter(jit[j], vals[j], s=30, alpha=0.85, color=color, marker=marker)
            # pooled mean/SD across hemis & subjects (all pair observations)
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            plt.scatter([x], [mu], s=MEAN_S, color="black", zorder=5)
            plt.errorbar([x], [mu], yerr=[[sd],[sd]], fmt="none",
                         ecolor="black", elinewidth=2, capsize=6, capthick=2)
    # legend & axes
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', label='Left (L)', markerfacecolor='black', markersize=7),
        Line2D([0],[0], marker='x', color='black', label='Right (R)', markersize=7, linestyle='None')
    ]
    plt.legend(handles=legend_elems, title="Hemisphere", loc="best", frameon=True)
    plt.xticks(xs, ["/".join(map(str, lab)) for lab in labels])
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# Consistency
labels_c = [(ver,) for ver in version_dirs.keys()]
scatter_block(
    df_sub=df_all[df_all["measure"]=="consistency"].rename(columns={"value":"metric"}),
    value_col="metric",
    labels=labels_c,
    group_cols=["version"],
    title=f"Consistency (per-pair within-subject) — {dataset}",
    out_png=str(OUTDIR / f"consistency_scatter_{dataset}.png"),
)
print(f"Saved: {OUTDIR / f'consistency_scatter_{dataset}.png'}")

# Identifiability
labels_i = [(ver,) for ver in version_dirs.keys()]
scatter_block(
    df_sub=df_all[df_all["measure"]=="identifiability"].rename(columns={"value":"metric"}),
    value_col="metric",
    labels=labels_i,
    group_cols=["version"],
    title=f"Identifiability ((C-B)/C, per pair; B = Fisher-z mean vs all other sessions) — {dataset}",
    out_png=str(OUTDIR / f"identifiability_scatter_{dataset}.png"),
)
print(f"Saved: {OUTDIR / f'identifiability_scatter_{dataset}.png'}")

# ---------------------------------------
# Correlation matrices (Left & Right combined, using OVERALL means)
# Rows/cols: [all Left subjects, then all Right subjects]
# ---------------------------------------

def plot_corr_matrix_LR(ax, corr, n_left, title):
    im = ax.imshow(corr, vmin=0.9, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    if n_left > 0 and n_left < corr.shape[0]:
        ax.axhline(n_left - 0.5, color="k", lw=0.8)
        ax.axvline(n_left - 0.5, color="k", lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im

def collect_subj_overall_means_for(ver, hemi):
    # reuse precomputed overall_means
    means = overall_means[hemi][ver]
    ids, mats = [], []
    # enforce equal length for stacking
    expected_len = None
    for subj in subjects:
        v = means.get(subj)
        if v is not None:
            expected_len = v.size
            break
    if expected_len is None:
        return [], []
    for subj in subjects:
        v = means.get(subj)
        if v is not None and v.size == expected_len:
            ids.append(subj); mats.append(v)
    return ids, mats

for ver in version_dirs.keys():
    ids_L, means_L = collect_subj_overall_means_for(ver, "L")
    ids_R, means_R = collect_subj_overall_means_for(ver, "R")
    if len(means_L) + len(means_R) < 2:
        continue
    blocks = []
    if len(means_L) > 0:
        blocks.append(np.vstack(means_L))
    n_left = blocks[0].shape[0] if blocks else 0
    if len(means_R) > 0:
        blocks.append(np.vstack(means_R))
    M = np.vstack(blocks) if blocks else None
    if M is None or M.shape[0] < 2:
        continue
    corr = np.corrcoef(M)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5), constrained_layout=True)
    im = plot_corr_matrix_LR(ax, corr, n_left, f"{ver}: L↔R combined (overall means)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_ylabel('r', rotation=0, labelpad=10)
    out_path = OUTDIR / f"corrmat_{ver}_{dataset}.png"
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"Saved: {out_path}")

# ---------------------------------------
# ANOVAs 
# ---------------------------------------

from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf

def run_rm_version_only(df_all, measure_name: str, subject_col="subject_id"):
    # 1) subset and clean
    df_m = df_all[df_all["measure"] == measure_name].dropna(subset=["value"]).copy()
    if df_m.empty:
        print(f"\n[RM-ANOVA] {measure_name}: no data")
        return

    # 2) collapse across hemisphere *and* pair (exclude hemi from stats; 1 obs per subject×version)
    #    If you want to keep pairs separate, you'll need a balanced within-subject factor for pairs.
    collapse_cols = [subject_col, "version"]
    df_cell = (
        df_m.groupby(collapse_cols, dropna=False, as_index=False)["value"]
            .mean()
            .rename(columns={"value": "cell_mean"})
    ).dropna(subset=["cell_mean"])

    # 3) keep only subjects with all version levels present (balanced RM)
    ver_levels = sorted(df_cell["version"].unique())
    need = len(ver_levels)
    subj_counts = df_cell.groupby(subject_col).size()
    complete_subjects = subj_counts[subj_counts == need].index.tolist()
    df_bal = df_cell[df_cell[subject_col].isin(complete_subjects)].copy()

    print(f"\nRepeated-measures ANOVA for {measure_name} (within: version; hemi excluded)")
    print(f"Subjects total: {df_cell[subject_col].nunique()} | complete for RM: {len(complete_subjects)}")
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
        except Exception as e:
            print(f"[Warning] AnovaRM failed ({e}). Falling back to MixedLM.")
            # Mixed model fallback: random intercept per subject
            df_bal["version"] = df_bal["version"].astype("category")
            md = smf.mixedlm("cell_mean ~ C(version)", df_bal, groups=df_bal[subject_col])
            mdf = md.fit(method="lbfgs", reml=True, maxiter=200, disp=False)
            print(mdf.summary())
    else:
        print("Not enough complete subjects for RM-ANOVA. Using MixedLM (random intercept).")
        df_cell["version"] = df_cell["version"].astype("category")
        md = smf.mixedlm("cell_mean ~ C(version)", df_cell, groups=df_cell[subject_col])
        mdf = md.fit(method="lbfgs", reml=True, maxiter=200, disp=False)
        print(mdf.summary())

# ----- replace your previous ANOVA loop with this -----
for measure in ["consistency", "identifiability"]:
    run_rm_version_only(df_all, measure_name=measure, subject_col="subject_id")
