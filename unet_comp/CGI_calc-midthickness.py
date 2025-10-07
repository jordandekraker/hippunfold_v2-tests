#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, re
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
from typing import Optional, Tuple, List

# ---------------------------
# Config (edit if needed)
# ---------------------------
DATASETS   = ["PNI", "MICs", "bMICs"]
CONDITIONS = ["T1w", "synthseg_v0.2"]
HEMIS      = ["L", "R"]
FEATURES   = ["midthickness"]

DEN    = "8k"
LABELS = ["hipp", "dentate"]   # <-- NEW: concatenate these (in this order)
SUBDIR = "surf"  # geometry lives in *.surf.gii

# Roots to micapipe derivatives where the ANTs affines live
MICAPIPE_ROOTS = {
    "PNI":   "/data/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0",
    "MICs":  "/data/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0",
    "bMICs": "/data/mica3/BIDS_bMICs/derivatives/micapipe_v0.2.0",
}

SUBJECT_LIST_FILES = {
    "MICs":  "participants-MICs.txt",
    "PNI":   "participants-PNI.txt",
    "bMICs": "participants-bMICs.txt",
}

MAX_SESSIONS = 5
RNG_SEED     = 42
JITTER_STD   = 0.035
MEAN_S       = 140

# ---------------------------
# Read participant lists (row aligned across datasets)
# ---------------------------
tables = {}
for ds, fn in SUBJECT_LIST_FILES.items():
    if os.path.exists(fn):
        with open(fn) as f:
            tables[ds] = [ln.strip() for ln in f if ln.strip()]
    else:
        tables[ds] = []
max_subject_rows = max([len(v) for v in tables.values()] + [0])

# ---------------------------
# Utilities: load ANTs 0GenericAffine.mat and apply to Nx3 points
# ---------------------------
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
    try:
        T = sitk.AffineTransform(T) if not isinstance(T, sitk.AffineTransform) else T
    except Exception:
        return None
    if T.GetDimension() != 3:
        return None
    if invert:
        try:
            T = T.GetInverse()
        except Exception:
            return None

    M = np.array(T.GetMatrix(), dtype=float).reshape(3, 3)
    c = np.array(T.GetCenter(), dtype=float)
    t = np.array(T.GetTranslation(), dtype=float)

    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    A = np.eye(4, dtype=float)
    A[:3, :3] = R
    A[:3, 3]  = t + c - R @ c
    return A

def apply_affine(points_xyz: np.ndarray, A: np.ndarray) -> np.ndarray:
    """points_xyz: (N,3). A: (4,4)."""
    N = points_xyz.shape[0]
    homo = np.c_[points_xyz, np.ones((N, 1), dtype=points_xyz.dtype)]
    out = (A @ homo.T).T[:, :3]
    return out

def _first_existing_path(base_dir: str, sub: str, ses: str, hemi: str, label: str, feat: str) -> Optional[str]:
    """Find one existing file path, or None."""
    patt = os.path.join(
        base_dir, f"sub-{sub}", f"ses-{ses}", SUBDIR,
        f"sub-{sub}_ses-{ses}_hemi-{hemi}_space-T1w_den-{DEN}_label-{label}_{feat}.surf.gii"
    )
    cands = sorted(glob.glob(patt))
    return cands[0] if cands else None

def _load_coords(path: str) -> Optional[np.ndarray]:
    try:
        g = nib.load(path)
        coords = np.asarray(g.darrays[0].data, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            return None
        return coords
    except Exception:
        return None

def probe_vector_length(datasets: List[str], conditions: List[str], hemis: List[str]) -> int:
    """
    Probe the filesystem to determine the flattened length F of the concatenated vector:
    sum over labels × features of (num_vertices * 3).
    Fallback: assumes 8k per label if nothing is found.
    """
    for ds in datasets:
        for cond in conditions:
            base_dir = f"hippunfold_{ds}_{cond}"
            # walk a couple of subjects/sessions to find a hit
            sub_dirs = sorted(glob.glob(os.path.join(base_dir, "sub-*")))
            for sub_dir in sub_dirs[:10]:
                sub = os.path.basename(sub_dir).split("-")[-1]
                ses_dirs = sorted(glob.glob(os.path.join(sub_dir, "ses-*")))
                for ses_dir in ses_dirs[:10]:
                    ses = os.path.basename(ses_dir).split("-")[-1]
                    for hemi in hemis:
                        total = 0
                        ok = True
                        for label in LABELS:
                            for feat in FEATURES:
                                p = _first_existing_path(base_dir, sub, ses, hemi, label, feat)
                                if p is None:
                                    ok = False
                                    break
                                coords = _load_coords(p)
                                if coords is None:
                                    ok = False
                                    break
                                total += coords.shape[0] * 3
                            if not ok:
                                break
                        if ok and total > 0:
                            return total
    # fallback: old 8k assumption, doubled for two labels
    nV_default = 8192
    F_fallback = int(len(FEATURES) * 3 * nV_default * len(LABELS))
    print(f"[INFO] probe_vector_length: using fallback F={F_fallback} (no files found during probe)")
    return F_fallback

# ---------------------------
# Helper to load & concat one session vector (returns NaN vector when missing)
# ---------------------------
def load_concat_vector(base_dir: str, ds: str, sub: str, ses: str, hemi: str, F_target: int) -> np.ndarray:
    vecs = []
    # For bMICs, use MICs affines
    ds_for_affine = "MICs" if ds == "bMICs" else ds

    # Build transform path (micapipe derivatives)
    xfm_root = MICAPIPE_ROOTS.get(ds_for_affine, "")
    mat_path = os.path.join(
        xfm_root,
        f"sub-{sub}",
        f"ses-{ses}",
        "xfm",
        f"sub-{sub}_ses-{ses}_from-nativepro_brain_to-MNI152_2mm_mode-image_desc-SyN_0GenericAffine.mat",
    )
    A = read_ants_rigid_4x4(mat_path)
    if A is None:
        return np.full((F_target,), np.nan, dtype=np.float64)

    # Concatenate in a fixed, reproducible order: LABELS (hipp, dentate) × FEATURES
    for label in LABELS:
        for feat in FEATURES:
            p = _first_existing_path(base_dir, sub, ses, hemi, label, feat)
            if p is None:
                return np.full((F_target,), np.nan, dtype=np.float64)
            coords = _load_coords(p)
            if coords is None:
                return np.full((F_target,), np.nan, dtype=np.float64)

            # ---- APPLY TRANSFORM to MNI152 (linear) ----
            coords_mni = apply_affine(coords, A)
            coords_norm = coords_mni - np.nanmean(coords_mni, axis=0, keepdims=True)
            if hemi == "L":
                coords_norm[:, 1] *= -1
            arr = coords_norm.reshape(-1)
            vecs.append(arr)

    cat = np.concatenate(vecs, axis=0) if vecs else np.empty((0,), dtype=np.float64)
    # Pad/truncate to target length to keep array shapes consistent
    out = np.full((F_target,), np.nan, dtype=np.float64)
    n = min(F_target, cat.size)
    if n > 0:
        out[:n] = cat[:n]
    return out

# ===========================
# Allocate, populate, and compute metrics
# ===========================
# 1) Probe F dynamically (total concatenated feature length)
F = probe_vector_length(DATASETS, CONDITIONS, HEMIS)
print(f"[INFO] Using F={F} features per hemi (concat of {LABELS} × {FEATURES} × xyz)")

# 2) Allocate big tensor: X[d,c,h,s,t,f] filled with NaN
X = np.full(
    (len(DATASETS), len(CONDITIONS), len(HEMIS), max_subject_rows, MAX_SESSIONS, F),
    np.nan, dtype=np.float64
)

# 3) Populate X from disk
for d_i, ds in enumerate(DATASETS):
    subj_ids = tables.get(ds, [])
    for c_i, cond in enumerate(CONDITIONS):
        base_dir = f"hippunfold_{ds}_{cond}"
        for h_i, hemi in enumerate(HEMIS):
            for s_i in range(max_subject_rows):
                sub = subj_ids[s_i] if s_i < len(subj_ids) else None
                if sub is None:
                    continue
                ses_dirs = glob.glob(os.path.join(base_dir, f"sub-{sub}", "ses-*"))
                ses_list = sorted([os.path.basename(p).split("-")[-1] for p in ses_dirs if os.path.isdir(p)])
                if not ses_list:
                    continue
                for t_i, ses in enumerate(ses_list[:MAX_SESSIONS]):
                    vec = load_concat_vector(base_dir, ds, sub, ses, hemi, F)
                    X[d_i, c_i, h_i, s_i, t_i, :] = vec

# ---------------------------
# Block 1: Consistency (within-subject across sessions)
# ---------------------------
def pearsonr_nan(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    a2 = a[m]; b2 = b[m]
    sa = a2.std(); sb = b2.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a2, b2)[0, 1])

def mae_nan(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 1:
        return np.nan
    a2 = a[m]; b2 = b[m]
    return float(np.mean(np.abs(a2 - b2)))

rows_consistency = []
for d_i, ds in enumerate(DATASETS):
    for c_i, cond in enumerate(CONDITIONS):
        for h_i, hemi in enumerate(HEMIS):
            for s_i in range(max_subject_rows):
                sess_vecs = []
                for t_i in range(MAX_SESSIONS):
                    v = X[d_i, c_i, h_i, s_i, t_i, :]
                    if np.all(np.isnan(v)):
                        continue
                    sess_vecs.append(v)
                if len(sess_vecs) < 2:
                    cons = np.nan
                else:
                    cors = []
                    for i in range(len(sess_vecs)):
                        for j in range(i+1, len(sess_vecs)):
                            cors.append(pearsonr_nan(sess_vecs[i], sess_vecs[j]))
                    cons = np.nanmean(cors) if len(cors) else np.nan
                rows_consistency.append([ds, cond, hemi, s_i, cons])

df_cons = pd.DataFrame(rows_consistency, columns=["dataset","condition","hemi","subject_row","consistency"])

# ---------------------------
# Block 2: Identifiability (normalized separation)
# ---------------------------
MEANS = np.nanmean(X, axis=4)  # session-mean: (D,C,H,S,F)

rows_ident = []
for d_i, ds in enumerate(DATASETS):
    for c_i, cond in enumerate(CONDITIONS):
        for h_i, hemi in enumerate(HEMIS):
            M = MEANS[d_i, c_i, h_i, :, :]  # (S,F)
            for s_i in range(max_subject_rows):
                m_this = M[s_i, :]
                if np.all(np.isnan(m_this)):
                    ident = np.nan
                else:
                    cors = []
                    for s_j in range(max_subject_rows):
                        if s_j == s_i:
                            continue
                        m_other = M[s_j, :]
                        if np.all(np.isnan(m_other)):
                            continue
                        cors.append(pearsonr_nan(m_this, m_other))
                    between = np.nanmean(cors) if len(cors) else np.nan
                    cons_row = df_cons[
                        (df_cons["dataset"]==ds) &
                        (df_cons["condition"]==cond) &
                        (df_cons["hemi"]==hemi) &
                        (df_cons["subject_row"]==s_i)
                    ]
                    cons = cons_row["consistency"].values[0] if len(cons_row)==1 else np.nan
                    ident = ((cons - between) / cons ) if (not np.isnan(between) and not np.isnan(cons) and cons != 0) else np.nan
                rows_ident.append([ds, cond, hemi, s_i, ident])

df_ident = pd.DataFrame(rows_ident, columns=["dataset","condition","hemi","subject_row","identifiability"])

# ---------------------------
# Block 3: Generalizability (pairwise: PNI–MICs, MICs–bMICs, bMICs–PNI)
# ---------------------------
PAIR_NAMES = [("PNI","MICs"), ("MICs","bMICs"), ("bMICs","PNI")]
rows_gen = []
ds_to_idx = {ds:i for i,ds in enumerate(DATASETS)}

for c_i, cond in enumerate(CONDITIONS):
    for h_i, hemi in enumerate(HEMIS):
        for s_i in range(max_subject_rows):
            for (a_name, b_name) in PAIR_NAMES:
                a = ds_to_idx.get(a_name, None)
                b = ds_to_idx.get(b_name, None)
                if a is None or b is None:
                    continue
                va = MEANS[a, c_i, h_i, s_i, :]
                vb = MEANS[b, c_i, h_i, s_i, :]
                if np.all(np.isnan(va)) or np.all(np.isnan(vb)):
                    corr = np.nan
                else:
                    corr = pearsonr_nan(va, vb)
                rows_gen.append([f"{a_name}-{b_name}", cond, hemi, s_i, corr])

df_gen = pd.DataFrame(
    rows_gen,
    columns=["dataset_pair","condition","hemi","subject_row","generalizability"]
)

# ---------------------------
# Save combined CSV
# ---------------------------
df_all = pd.concat([
    df_cons.assign(measure="consistency",       value=df_cons["consistency"])[["dataset","condition","hemi","subject_row","measure","value"]],
    df_ident.assign(measure="identifiability",  value=df_ident["identifiability"])[["dataset","condition","hemi","subject_row","measure","value"]],
    df_gen.assign(measure="generalizability",   value=df_gen["generalizability"])[["dataset_pair","condition","hemi","subject_row","measure","value"]],
], ignore_index=True)

df_all.to_csv("metrics_per_subject.csv", index=False)
print("Done. Wrote metrics_per_subject.csv")
