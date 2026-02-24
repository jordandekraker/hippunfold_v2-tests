#!/usr/bin/env python3
"""
qc_coronal_grid_autodiscover.py

Autodiscover degradation levels under OUT_ROOT and save a coronal QC grid for the FIRST subject.

Grid:
  rows = noise levels (none + all discovered noise fracs)
  cols = blur levels  (none + all discovered blur sigmas)

Cells:
  (0,0) uses the clean TEMPLATE_IN_PATH (for that subject)
  noise-only uses OUT_ROOT/<noise_tag>
  blur-only uses OUT_ROOT/<blur_tag>
  blur+noise uses OUT_ROOT/<blurnoise_tag> (autodiscovered, all combos)

Writes:
  OUT_ROOT/_qc/<sub>_coronal_grid.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# =============================================================================
# USER EDIT ZONE
# =============================================================================

OUT_ROOT = Path("/host/bb-comp/tank/data/BIDS_PNI_degraded")

TEMPLATE_IN_PATH = Path(
    "/host/bb-comp/tank/data/BIDS_PNI/derivatives/hippunfold_v2.0.0/"
    "sub-PNC010/ses-01/anat/sub-PNC010_ses-01_desc-preproc_T1w.nii.gz"
)

SUBS: List[str] = [
    "sub-PNC003",
    # "sub-PNC007",
]

DISPLAY_PCTS = (2.0, 98.0)
CORONAL_Y_INDEX: int | None = None

# =============================================================================


def build_clean_in_path_for_sub(sub_tag: str) -> Path:
    p = Path(str(TEMPLATE_IN_PATH).replace("sub-PNC010", sub_tag))
    p = Path(
        str(p).replace(
            "sub-PNC010_ses-01_desc-preproc_T1w.nii.gz",
            f"{sub_tag}_ses-01_desc-preproc_T1w.nii.gz",
        )
    )
    return p


def degraded_nii_path(sub_tag: str, tag_dir: Path) -> Path:
    # OUT_ROOT/<tag>/sub-XXXX/anat/sub-XXXX_T1w.nii.gz
    return tag_dir / sub_tag / "anat" / f"{sub_tag}_T1w.nii.gz"


def load_coronal_slice(path: Path, y_index: int | None) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got {data.shape} for {path}")

    y = (data.shape[1] // 2) if y_index is None else int(y_index)
    y = max(0, min(y, data.shape[1] - 1))

    sl = data[:, y, :]
    sl = np.rot90(sl)

    # --- NEW: crop top and bottom 25% ---
    h = sl.shape[0]
    crop = int(0.25 * h)
    sl = sl[crop : h - crop, :]

    return sl


def robust_vmin_vmax(arr: np.ndarray, pcts=(2.0, 98.0)) -> Tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(vals, pcts)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return float(np.nanmin(vals)), float(np.nanmax(vals))
    return float(vmin), float(vmax)


def parse_noise_frac(tag: str) -> Optional[float]:
    # noise-L1_frac0p020
    m = re.search(r"frac([0-9]+p[0-9]+)", tag)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def parse_blur_sig(tag: str) -> Optional[Tuple[float, float]]:
    # blur-L1_sig0p50-0p80
    m = re.search(r"sig([0-9]+p[0-9]+)-([0-9]+p[0-9]+)", tag)
    if not m:
        return None
    return (float(m.group(1).replace("p", ".")), float(m.group(2).replace("p", ".")))


def main() -> None:
    if not SUBS:
        raise SystemExit("SUBS is empty. Paste your subject list and keep the first one.")
    sub_tag = SUBS[0]

    qc_outdir = Path("plots")
    qc_outdir.mkdir(parents=True, exist_ok=True)

    # ---- discover directories
    dirs = [p for p in OUT_ROOT.iterdir() if p.is_dir() and p.name != "_qc"]

    noise_dirs = sorted([p for p in dirs if p.name.startswith("noise-L")])
    blur_dirs = sorted([p for p in dirs if p.name.startswith("blur-L")])
    blurnoise_dirs = sorted([p for p in dirs if p.name.startswith("blurnoise-")])

    print(f"[INFO] OUT_ROOT: {OUT_ROOT}")
    print(f"[INFO] Found dirs: noise={len(noise_dirs)} blur={len(blur_dirs)} blurnoise={len(blurnoise_dirs)}")
    if len(noise_dirs) == 0:
        print("[WARN] No noise-L* directories found. Listing first 30 dirs under OUT_ROOT:")
        for p in sorted(dirs)[:30]:
            print("  ", p.name)

    # map levels by parsed parameters
    noise_levels: List[Tuple[float, Path]] = []
    for p in noise_dirs:
        frac = parse_noise_frac(p.name)
        if frac is not None:
            noise_levels.append((frac, p))
    noise_levels.sort(key=lambda x: x[0])

    blur_levels: List[Tuple[Tuple[float, float], Path]] = []
    for p in blur_dirs:
        sig = parse_blur_sig(p.name)
        if sig is not None:
            blur_levels.append((sig, p))
    blur_levels.sort(key=lambda x: (x[0][0], x[0][1]))

    # build combo lookup: (noise_frac, blur_sig) -> dir
    combo_lookup: Dict[Tuple[float, Tuple[float, float]], Path] = {}
    for p in blurnoise_dirs:
        frac = parse_noise_frac(p.name)
        sig = parse_blur_sig(p.name)
        if frac is not None and sig is not None:
            combo_lookup[(frac, sig)] = p

    # ---- load clean slice for scaling
    clean_path = build_clean_in_path_for_sub(sub_tag)
    if not clean_path.exists():
        raise SystemExit(f"Clean input not found: {clean_path}")

    clean_sl = load_coronal_slice(clean_path, CORONAL_Y_INDEX)
    vmin, vmax = robust_vmin_vmax(clean_sl, DISPLAY_PCTS)

    # ---- grid dims (include "none")
    n_rows = 1 + len(noise_levels)
    n_cols = 1 + len(blur_levels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), dpi=150)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    col_titles = ["blur: none"] + [f"blur σ∈[{s[0]:.2f},{s[1]:.2f}]" for (s, _) in blur_levels]
    row_titles = ["noise: none"] + [f"noise frac={f:.3f}" for (f, _) in noise_levels]

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.axis("off")

            if i == 0 and j == 0:
                sl = clean_sl
            elif i > 0 and j == 0:
                frac, ndir = noise_levels[i - 1]
                nii = degraded_nii_path(sub_tag, ndir)
                if not nii.exists():
                    ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=14)
                    ax.set_title(ndir.name, fontsize=8)
                    continue
                sl = load_coronal_slice(nii, CORONAL_Y_INDEX)
            elif i == 0 and j > 0:
                sig, bdir = blur_levels[j - 1]
                nii = degraded_nii_path(sub_tag, bdir)
                if not nii.exists():
                    ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=14)
                    ax.set_title(bdir.name, fontsize=8)
                    continue
                sl = load_coronal_slice(nii, CORONAL_Y_INDEX)
            else:
                frac, _ = noise_levels[i - 1]
                sig, _ = blur_levels[j - 1]
                cdir = combo_lookup.get((frac, sig), None)
                if cdir is None:
                    ax.text(0.5, 0.5, "NO COMBO DIR", ha="center", va="center", fontsize=12)
                    continue
                nii = degraded_nii_path(sub_tag, cdir)
                if not nii.exists():
                    ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=14)
                    ax.set_title(cdir.name, fontsize=8)
                    continue
                sl = load_coronal_slice(nii, CORONAL_Y_INDEX)

            ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)

            if i == 0:
                ax.set_title(col_titles[j], fontsize=10)
            if j == 0:
                ax.text(
                    -0.02, 0.5, row_titles[i],
                    transform=ax.transAxes,
                    ha="right", va="center",
                    rotation=90, fontsize=10
                )

    fig.suptitle(f"{sub_tag} coronal QC grid (rows=noise, cols=blur)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = qc_outdir / f"{sub_tag}_coronal_grid.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote: {out_png}")


if __name__ == "__main__":
    main()
