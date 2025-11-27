#!/usr/bin/env python3
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from nibabel.nifti1 import intent_codes
import SimpleITK as sitk
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
import pyvista as pv
from datetime import datetime

# ========================
# Configuration
# ========================
DATASETS = ["BIDS_MICs", "BIDS_PNI"]
LABELS = ["hipp", "dentate"]
DATASET_SESSION_CONFIG = {
    "BIDS_MICs": [("ses-01", "ses-02")],
    "BIDS_PNI": [("ses-01", "ses-02"), ("ses-02", "ses-03"), ("ses-03", "ses-01")]
}
HEMI_ORDER = ["L", "R"]
ALLOWED_DEN = {"8k", "0p5mm"}
OUTDIR = Path("plots_synth")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Plot settings
JITTER_STD = 0.035
MEAN_S = 140
PALETTE = "jet"
HEMI_MARK = {"L": "o", "R": "x"}
RNG = np.random.default_rng(42)

# Filtering
CONSISTENCY_THRESHOLD = 0.85  # exclude subject/hemi/version if mean consistency < threshold

# ========================
# Logging
# ========================
class Logger:
    """Redirect print statements to both console and log file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = OUTDIR / f"analysis_log_{timestamp}.txt"
logger = Logger(log_file)
sys.stdout = logger

print(f"Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_file}")
print("="*60)

# ========================
# Paths & I/O helpers
# ========================

def get_version_dirs(dataset):
    """Return only HippUnfold v2 versions: default and synthlayer."""
    base = Path(f"/export03/data/opt/hippunfold_v2-tests/v1-v2_comp/{dataset}")
    return {
        "default": base / "hippunfold_v2.0.0",
        "synthlayer": base / "hippunfold_v2.0.0_synthlayer",
    }


def micapipe_root_for_affine(ds):
    if ds == "BIDS_bMICs":
        return "/data/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0"
    return f"/data/mica3/{ds}/derivatives/micapipe_v0.2.0"


def read_ants_rigid_4x4(mat_path, invert=True):
    """Read ANTS transform and convert to 4x4 affine matrix."""
    try:
        T = sitk.ReadTransform(mat_path)
        T = sitk.AffineTransform(T) if not isinstance(T, sitk.AffineTransform) else T
        if T.GetDimension() != 3:
            return None
        if invert:
            T = T.GetInverse()
        M = np.array(T.GetMatrix(), dtype=float).reshape(3, 3)
        c = np.array(T.GetCenter(), dtype=float)
        t = np.array(T.GetTranslation(), dtype=float)
        # Extract rotation via SVD
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1.0
            R = U @ Vt
        A = np.eye(4, dtype=float)
        A[:3, :3] = R
        A[:3, 3] = t + c - R @ c
        return A
    except Exception:
        return None


def apply_affine(points_xyz, A):
    """Apply 4x4 affine transformation to Nx3 points."""
    N = points_xyz.shape[0]
    homo = np.c_[points_xyz, np.ones((N, 1), dtype=points_xyz.dtype)]
    return (A @ homo.T).T[:, :3]


def affine_for(ds, subj, ses):
    """Get affine transform for subject/session."""
    root = micapipe_root_for_affine(ds)
    mat_path = os.path.join(
        root, f"sub-{subj}", f"{ses}", "xfm",
        f"sub-{subj}_{ses}_from-nativepro_brain_to-MNI152_0.8mm_mode-image_desc-SyN_0GenericAffine.mat",
    )
    return read_ants_rigid_4x4(mat_path, invert=True)


def load_gifti_vertices(filepath):
    """Load vertices from GIFTI surface file."""
    gii = nib.load(str(filepath))
    verts = next(arr.data for arr in gii.darrays if arr.intent == intent_codes['NIFTI_INTENT_POINTSET'])
    return np.asarray(verts, dtype=np.float64)


def load_gifti_verts_faces(filepath):
    """Load both vertices and faces from GIFTI surface file."""
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


def find_surface(root_dir, subj, ses, hemi, label):
    """Find surface file for given parameters."""
    surf_dir = root_dir / f"sub-{subj}" / f"{ses}" / "surf"
    patterns = [
        f"sub-{subj}_{ses}_hemi-{hemi}_space-T1w_den-8k_label-{label}_midthickness.surf.gii",
        f"sub-{subj}_{ses}_hemi-{hemi}_space-T1w_den-0p5mm_label-{label}_midthickness.surf.gii",
    ]
    for pat in patterns:
        matches = list(surf_dir.glob(pat))
        if matches:
            return matches[0]
    glob_pat = surf_dir / f"sub-{subj}_{ses}_hemi-{hemi}_space-T1w_den-*_label-{label}_midthickness.surf.gii"
    matches = sorted(glob_pat.parent.glob(glob_pat.name))
    matches = [m for m in matches if any(f"den-{d}" in m.name for d in ALLOWED_DEN)]
    return matches[0] if matches else None


def pearsonr_nan(a, b):
    """Pearson correlation ignoring NaNs."""
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    a2, b2 = a[m], b[m]
    sa, sb = a2.std(), b2.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a2, b2)[0, 1])


def get_session_vector(ver_root, ds, subj, ses, hemi):
    """Get normalized vertex vector for a session."""
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
        # Transform to MNI and normalize
        verts_mni = apply_affine(verts, A)
        verts_norm = verts_mni - np.mean(verts_mni, axis=0, keepdims=True)
        if hemi == "L":
            verts_norm[:, 0] *= -1
        parts.append(verts_norm.reshape(-1))
    return np.concatenate(parts, axis=0)


def collect_all_sessions(version_dirs, ds, hemi, session_pairs):
    """Collect all session vectors for all subjects."""
    all_sessions = sorted({s for pair in session_pairs for s in pair})
    out = {}
    with open(f"{ds}/participants.txt", "r") as f:
        subjects = sorted(set(f.read().split()))
    for ver, root in version_dirs.items():
        subj_sess = {}
        for subj in subjects:
            vecs = []
            for ses in all_sessions:
                v = get_session_vector(root, ds, subj, ses, hemi)
                if v is not None:
                    vecs.append(v)
            subj_sess[subj] = vecs
        out[ver] = subj_sess
    return out


def collect_overall_means(version_dirs, ds, hemi, session_pairs):
    """Collect mean vectors across all sessions for each subject."""
    all_sessions = sorted({s for pair in session_pairs for s in pair})
    out = {}
    with open(f"{ds}/participants.txt", "r") as f:
        subjects = sorted(set(f.read().split()))
    for ver, root in version_dirs.items():
        expected_len = None
        for subj in subjects:
            vecs = []
            for ses in all_sessions:
                v = get_session_vector(root, ds, subj, ses, hemi)
                if v is not None:
                    vecs.append(v)
            if vecs and len({vv.size for vv in vecs}) == 1:
                expected_len = vecs[0].size
                break
        subj_means = {}
        for subj in subjects:
            vecs = []
            for ses in all_sessions:
                v = get_session_vector(root, ds, subj, ses, hemi)
                if v is not None:
                    vecs.append(v)
            vecs = [v for v in vecs if expected_len is None or v.size == expected_len]
            subj_means[subj] = np.nanmean(np.vstack(vecs), axis=0) if vecs and expected_len else None
        out[ver] = subj_means
    return out


def scatter_block(df_sub, value_col, labels, group_cols, title, out_png, color_lookup):
    """Create scatter plot with means and error bars."""
    plt.figure(figsize=(3, 4))
    xs = np.arange(len(labels))
    for x, lab in zip(xs, labels):
        mask = np.ones(len(df_sub), dtype=bool)
        for col, val in zip(group_cols, lab):
            mask &= (df_sub[col] == val)
        cur = df_sub.loc[mask, ["subject_id", "hemi", value_col]].dropna(subset=[value_col])
        if len(cur) == 0:
            continue
        vals = cur[value_col].astype(float).to_numpy()
        subs = cur["subject_id"].astype(str).to_numpy()
        hemis = cur["hemi"].astype(str).tolist()
        jit = RNG.normal(loc=x, scale=JITTER_STD, size=len(vals))
        for j in range(len(vals)):
            marker = HEMI_MARK.get(hemis[j], "o")
            color = color_lookup.get(subs[j], "gray")
            if marker == "o":
                plt.scatter(jit[j], vals[j], s=30, alpha=0.85, color=color, marker=marker, linewidths=0)
            else:
                plt.scatter(jit[j], vals[j], s=30, alpha=0.85, color=color, marker=marker)
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        plt.scatter([x], [mu], s=MEAN_S, color="black", zorder=5)
        plt.errorbar([x], [mu], yerr=[[sd], [sd]], fmt="none",
                     ecolor="black", elinewidth=2, capsize=6, capthick=2)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Left (L)', markerfacecolor='black', markersize=7),
        Line2D([0], [0], marker='x', color='black', label='Right (R)', markersize=7, linestyle='None')
    ]
    plt.legend(handles=legend_elems, title="Hemisphere", loc="best", frameon=True)
    plt.xticks(xs, ["/".join(map(str, lab)) for lab in labels])
    plt.grid(alpha=0.2, axis="y")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_corr_matrix_LR(ax, corr, n_left, title):
    """Plot correlation matrix with L/R boundary."""
    im = ax.imshow(corr, vmin=0.9, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if n_left > 0 and n_left < corr.shape[0]:
        ax.axhline(n_left - 0.5, color="k", lw=0.8)
        ax.axvline(n_left - 0.5, color="k", lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def collect_subj_overall_means_for(ver, hemi, overall_means, subjects):
    """Collect overall means for a specific version and hemisphere."""
    means = overall_means[hemi][ver]
    ids, mats = [], []
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
            ids.append(subj)
            mats.append(v)
    return ids, mats


def run_rm_version_only(df_all, measure_name, subject_col="subject_id"):
    """Run repeated-measures ANOVA for version and hemisphere comparison."""
    df_m = df_all[df_all["measure"] == measure_name].dropna(subset=["value"]).copy()
    if df_m.empty:
        print(f"\n[RM-ANOVA] {measure_name}: no data")
        return
    df_cell = (
        df_m.groupby([subject_col, "version", "hemi"], dropna=False, as_index=False)["value"]
            .mean()
            .rename(columns={"value": "cell_mean"})
    ).dropna(subset=["cell_mean"])
    ver_levels = sorted(df_cell["version"].unique())
    hemi_levels = sorted(df_cell["hemi"].unique())
    need = len(ver_levels) * len(hemi_levels)
    subj_counts = df_cell.groupby(subject_col).size()
    complete_subjects = subj_counts[subj_counts == need].index.tolist()
    df_bal = df_cell[df_cell[subject_col].isin(complete_subjects)].copy()
    print(f"\nRepeated-measures ANOVA for {measure_name} (within: version, hemi)")
    print(f"Subjects total: {df_cell[subject_col].nunique()} | complete for RM: {len(complete_subjects)}")
    print(f"Version levels: {ver_levels}")
    print(f"Hemisphere levels: {hemi_levels}")
    if len(complete_subjects) >= 2:
        try:
            aov = AnovaRM(
                data=df_bal,
                depvar="cell_mean",
                subject=subject_col,
                within=["version", "hemi"]
            ).fit()
            print(aov.anova_table)
        except Exception as e:
            print(f"[Warning] AnovaRM failed ({e}). Falling back to MixedLM.")
            df_bal["version"] = df_bal["version"].astype("category")
            df_bal["hemi"] = df_bal["hemi"].astype("category")
            md = smf.mixedlm("cell_mean ~ C(version) * C(hemi)", df_bal, groups=df_bal[subject_col])
            mdf = md.fit(method="lbfgs", reml=True, maxiter=200, disp=False)
            print(mdf.summary())
    else:
        print("Not enough complete subjects for RM-ANOVA. Using MixedLM.")
        df_cell["version"] = df_cell["version"].astype("category")
        df_cell["hemi"] = df_cell["hemi"].astype("category")
        md = smf.mixedlm("cell_mean ~ C(version) * C(hemi)", df_cell, groups=df_cell[subject_col])
        mdf = md.fit(method="lbfgs", reml=True, maxiter=200, disp=False)
        print(mdf.summary())


def concat_two_meshes_to_polydata(path_a, path_b):
    """Concatenate two triangle meshes (A then B) into a single PolyData."""
    Va, Fa = load_gifti_verts_faces(path_a)
    Vb, Fb = load_gifti_verts_faces(path_b)
    V = np.vstack([Va, Vb])
    Fb_off = Fb + Va.shape[0]
    F = np.vstack([Fa, Fb_off])
    faces_vtk = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
    return pv.PolyData(V, faces_vtk)


def compute_cell_quality_for_session(ver_root, dataset, subj, ses, hemi):
    """Compute concatenated mesh cell quality for a given session."""
    surf_hipp = find_surface(ver_root, subj, ses, hemi, "hipp")
    surf_dentate = find_surface(ver_root, subj, ses, hemi, "dentate")
    if surf_hipp is None or surf_dentate is None:
        return None, None
    try:
        poly = concat_two_meshes_to_polydata(surf_hipp, surf_dentate)
        qmesh = poly.compute_cell_quality()
        q = np.asarray(qmesh["CellQuality"], dtype=float)
        if q.size == 0:
            return None, None
        mu = float(np.mean(q))
        sd = float(np.std(q, ddof=1)) if q.size > 1 else 0.0
        return mu, sd
    except Exception as e:
        print(f"[WARN] Failed computing cell quality for {subj} {ses} {hemi}: {e}")
        return None, None

# ========================
# MAIN LOOP
# ========================
for dataset in DATASETS:
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset}")
    print(f"{'='*60}")

    session_pairs = DATASET_SESSION_CONFIG[dataset]
    version_dirs = get_version_dirs(dataset)

    with open(f"{dataset}/participants.txt", "r") as f:
        subjects = sorted(set(f.read().split()))

    print(f"Subjects: {len(subjects)}")
    print(f"Session pairs: {session_pairs}")

    # Collect all data needed for identifiability computation
    session_bank = {hemi: collect_all_sessions(version_dirs, dataset, hemi, session_pairs)
                    for hemi in HEMI_ORDER}
    overall_means = {hemi: collect_overall_means(version_dirs, dataset, hemi, session_pairs)
                     for hemi in HEMI_ORDER}

    all_sessions = sorted({s for pair in session_pairs for s in pair})

    rows_cons = []
    rows_ident = []
    rows_quality = []

    for ver, ver_root in version_dirs.items():
        for hemi in HEMI_ORDER:
            for subj in subjects:
                # Quality per session
                for ses in all_sessions:
                    mu, sd = compute_cell_quality_for_session(ver_root, dataset, subj, ses, hemi)
                    if mu is not None:
                        rows_quality.append([ver, hemi, subj, ses, mu, sd])
                # Consistency & Identifiability per session pair
                for (sesA, sesB) in session_pairs:
                    vA = get_session_vector(ver_root, dataset, subj, sesA, hemi)
                    vB = get_session_vector(ver_root, dataset, subj, sesB, hemi)
                    if vA is None or vB is None or vA.size != vB.size or vA.size < 3:
                        continue
                    C_pair = pearsonr_nan(vA, vB)
                    if np.isnan(C_pair):
                        continue
                    rows_cons.append([ver, hemi, subj, f"{sesA}_{sesB}", float(C_pair)])
                    # Identifiability vs others
                    m_this_pair = np.nanmean(np.vstack([vA, vB]), axis=0)
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
                    B_pair = np.mean(cors) if cors else np.nan
                    I_pair = ((C_pair - B_pair) / C_pair) if (not np.isnan(B_pair) and C_pair != 0) else np.nan
                    rows_ident.append([ver, hemi, subj, f"{sesA}_{sesB}", I_pair])

    # Build raw DataFrames
    df_cons = pd.DataFrame(rows_cons, columns=["version", "hemi", "subject_id", "pair", "consistency"]) \
                 .sort_values(["version", "hemi", "subject_id", "pair"]) \
                 .reset_index(drop=True)
    df_ident = pd.DataFrame(rows_ident, columns=["version", "hemi", "subject_id", "pair", "identifiability"]) \
                  .sort_values(["version", "hemi", "subject_id", "pair"]) \
                  .reset_index(drop=True)
    df_quality = pd.DataFrame(rows_quality, columns=["version", "hemi", "subject_id", "session", "quality_mean", "quality_std"]) \
                    .sort_values(["version", "hemi", "subject_id", "session"]) \
                    .reset_index(drop=True)

    # Save raw metrics (no filtering) for provenance
    df_all_raw = pd.concat([
        df_cons.assign(measure="consistency", value=df_cons["consistency"])[
            ["version", "hemi", "subject_id", "pair", "measure", "value"]],
        df_ident.assign(measure="identifiability", value=df_ident["identifiability"])[
            ["version", "hemi", "subject_id", "pair", "measure", "value"]],
        df_quality.copy().assign(measure="cell_quality", pair=lambda d: d["session"], value=lambda d: d["quality_mean"]) [
            ["version", "hemi", "subject_id", "pair", "measure", "value"]]
    ], ignore_index=True)

    raw_csv_path = f"{OUTDIR}/metrics_per_subject_versions_{dataset}_pairs_raw.csv"
    df_all_raw.to_csv(raw_csv_path, index=False)
    print(f"Wrote raw metrics: {raw_csv_path}")

    # ========================
    # Consistency-based filtering
    # ========================
    if not df_cons.empty:
        grp = (df_cons.groupby(["version", "hemi", "subject_id"], as_index=False)["consistency"]
                    .mean()
                    .rename(columns={"consistency": "consistency_mean"}))
        good = grp[grp["consistency_mean"] >= CONSISTENCY_THRESHOLD]
        bad = grp[grp["consistency_mean"] < CONSISTENCY_THRESHOLD]
        print(f"Mean consistency filtering @ {CONSISTENCY_THRESHOLD:.2f}")
        print(f"  Good groups (kept): {len(good)} | Bad groups (excluded): {len(bad)}")
        # Keys for keeping
        keep_keys = set(map(tuple, good[["version", "hemi", "subject_id"]].to_records(index=False)))
        # Filter identifiability by matching version/hemi/subject
        df_ident_f = df_ident[[tuple(r) in keep_keys for r in df_ident[["version","hemi","subject_id"]].to_records(index=False)]]
        # Filter quality by matching version/hemi/subject
        df_quality_f = df_quality[[tuple(r) in keep_keys for r in df_quality[["version","hemi","subject_id"]].to_records(index=False)]]
    else:
        print("No consistency rows found; skipping filtering.")
        df_ident_f = df_ident.copy()
        df_quality_f = df_quality.copy()

    # Build filtered df_all for plots/stats (consistency unfiltered; others filtered)
    df_all = pd.concat([
        df_cons.assign(measure="consistency", value=df_cons["consistency"])[
            ["version", "hemi", "subject_id", "pair", "measure", "value"]],
        df_ident_f.assign(measure="identifiability", value=df_ident_f["identifiability"])[
            ["version", "hemi", "subject_id", "pair", "measure", "value"]],
        df_quality_f.copy().assign(measure="cell_quality", pair=lambda d: d["session"], value=lambda d: d["quality_mean"]) [
            ["version", "hemi", "subject_id", "pair", "measure", "value"]]
    ], ignore_index=True)

    filtered_csv_path = f"{OUTDIR}/metrics_per_subject_versions_{dataset}_pairs_filtered.csv"
    df_all.to_csv(filtered_csv_path, index=False)
    print(f"Wrote filtered metrics (for plots/stats): {filtered_csv_path}")

    # ========================
    # Plots
    # ========================
    all_subjects = sorted(df_all["subject_id"].dropna().unique())
    cmap = plt.get_cmap(PALETTE, max(len(all_subjects), 1))
    color_lookup = {sid: cmap(i % cmap.N) for i, sid in enumerate(all_subjects)}

    # Keep label order fixed: default, synthlayer
    version_labels = [("default",), ("synthlayer",)]

    # Consistency
    scatter_block(
        df_sub=df_all[df_all["measure"] == "consistency"].rename(columns={"value": "metric"}),
        value_col="metric",
        labels=version_labels,
        group_cols=["version"],
        title=f"Consistency — {dataset}",
        out_png=str(OUTDIR / f"consistency_scatter_{dataset}.png"),
        color_lookup=color_lookup,
    )
    print(f"Saved: {OUTDIR / f'consistency_scatter_{dataset}.png'}")

    # Identifiability (filtered)
    scatter_block(
        df_sub=df_all[df_all["measure"] == "identifiability"].rename(columns={"value": "metric"}),
        value_col="metric",
        labels=version_labels,
        group_cols=["version"],
        title=f"Identifiability (filtered) — {dataset}",
        out_png=str(OUTDIR / f"identifiability_scatter_{dataset}.png"),
        color_lookup=color_lookup,
    )
    print(f"Saved: {OUTDIR / f'identifiability_scatter_{dataset}.png'}")

    # Cell quality (filtered)
    scatter_block(
        df_sub=df_all[df_all["measure"] == "cell_quality"].rename(columns={"value": "metric"}),
        value_col="metric",
        labels=version_labels,
        group_cols=["version"],
        title=f"Cell Quality (hipp+dentate, filtered) — {dataset}",
        out_png=str(OUTDIR / f"cellquality_scatter_{dataset}.png"),
        color_lookup=color_lookup,
    )
    print(f"Saved: {OUTDIR / f'cellquality_scatter_{dataset}.png'}")

    # Correlation matrices per version
    for ver in ["default", "synthlayer"]:
        ids_L, means_L = collect_subj_overall_means_for(ver, "L", overall_means, subjects)
        ids_R, means_R = collect_subj_overall_means_for(ver, "R", overall_means, subjects)
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
        im = plot_corr_matrix_LR(ax, corr, n_left, f"{ver}: L↔R combined")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_ylabel('r', rotation=0, labelpad=10)
        out_path = OUTDIR / f"corrmat_{ver}_{dataset}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # ========================
    # Stats
    # ========================
    for measure in ["consistency", "identifiability", "cell_quality"]:
        run_rm_version_only(df_all, measure_name=measure, subject_col="subject_id")

print(f"\n{'='*60}")
print("All datasets processed successfully!")
print(f"{'='*60}")
print(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Close logger
logger.close()
sys.stdout = logger.terminal
