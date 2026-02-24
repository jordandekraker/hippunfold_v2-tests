#!/usr/bin/env python3
"""
Compute xyz coordinate correlations between degraded and clean hippunfold outputs.

For each degradation level:
  - stack label-hipp and label-dentate xyz coordinates
  - compute Pearson correlation vs clean
  - do separately for L and R hemispheres
  - n = (#subjects × 2 hemispheres)

Then display results in a 3x3 matrix (noise rows, blur columns).
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Dict


# =============================================================================
# USER EDIT ZONE
# =============================================================================

DEGRADED_ROOT = Path("/host/bb-comp/tank/data/BIDS_PNI_degraded")

CLEAN_ROOT = Path(
    "/host/bb-comp/tank/data/BIDS_PNI/derivatives/hippunfold_v2.0.0"
)

SUBS: list[str] = [
    "sub-PNC003",
    "sub-PNC007",
    "sub-PNC011",
    "sub-PNC014",
    "sub-PNC019",
    "sub-PNC022",
    "sub-PNC026",
    "sub-PNC020",
    "sub-PNC024",
    "sub-PNC027",
]

# =============================================================================

def load_xyz(path: Path) -> np.ndarray:
    gii = nib.load(str(path))

    for da in gii.darrays:
        intent = da.intent

        # POINTSET intent corresponds to vertex coordinates
        if intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
            return da.data.astype(np.float64)

    raise ValueError(f"No POINTSET (vertex) array found in {path}")


def stack_surfaces(root: Path, sub: str, hemi: str, ses: str | None) -> np.ndarray | None:
    """
    Strict loader. Returns stacked Nx3 coords, or None if missing.
    """
    if ses is None:
        surf_dir = root / sub / "surf"
        hipp_name = f"{sub}_hemi-{hemi}_space-T1w_den-8k_label-hipp_midthickness.surf.gii"
        dent_name = f"{sub}_hemi-{hemi}_space-T1w_den-8k_label-dentate_midthickness.surf.gii"
    else:
        surf_dir = root / sub / ses / "surf"
        hipp_name = f"{sub}_{ses}_hemi-{hemi}_space-T1w_den-8k_label-hipp_midthickness.surf.gii"
        dent_name = f"{sub}_{ses}_hemi-{hemi}_space-T1w_den-8k_label-dentate_midthickness.surf.gii"

    if not surf_dir.exists():
        print(f"[WARN] surf dir missing: {surf_dir}")
        return None

    hipp_files = list(surf_dir.glob(hipp_name))
    dent_files = list(surf_dir.glob(dent_name))

    if len(hipp_files) != 1 or len(dent_files) != 1:
        print(f"[WARN] Missing surface(s) for {sub} {hemi} in {surf_dir}")
        print(f"       hipp found: {len(hipp_files)}")
        print(f"       dent found: {len(dent_files)}")
        return None

    xyz_hipp = load_xyz(hipp_files[0])
    xyz_dent = load_xyz(dent_files[0])

    stacked = np.vstack([xyz_hipp, xyz_dent])
    if stacked.shape[1] != 3:
        print(f"[WARN] Bad coord shape for {sub} {hemi}: {stacked.shape}")
        return None

    return stacked


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)

    sa = a.std()
    sb = b.std()
    if sa == 0.0 or sb == 0.0:
        return float("nan")

    a = (a - a.mean()) / sa
    b = (b - b.mean()) / sb
    return float(np.mean(a * b))


def discover_levels():
    dirs = [p for p in DEGRADED_ROOT.iterdir() if p.is_dir() and p.name != "_qc"]

    noise_levels = []
    blur_levels = []
    combo_lookup: Dict[Tuple[int, int], Path] = {}

    for p in dirs:
        name = p.name

        if name.startswith("noise-L"):
            m = re.search(r"noise-L(\d+)", name)
            if m:
                noise_levels.append((int(m.group(1)), p))

        if name.startswith("blur-L"):
            m = re.search(r"blur-L(\d+)", name)
            if m:
                blur_levels.append((int(m.group(1)), p))

        if name.startswith("blurnoise-"):
            m = re.search(r"N(\d+)B(\d+)", name)
            if m:
                combo_lookup[(int(m.group(1)), int(m.group(2)))] = p

    noise_levels.sort()
    blur_levels.sort()

    return noise_levels, blur_levels, combo_lookup


def main():

    noise_levels, blur_levels, combo_lookup = discover_levels()

    n_rows = 1 + len(noise_levels)
    n_cols = 1 + len(blur_levels)

    results = np.full((n_rows, n_cols), np.nan, dtype=float)
    fails = np.zeros((n_rows, n_cols), dtype=int)
    nvalid = np.zeros((n_rows, n_cols), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            corrs = []
            fail_count = 0

            for sub in SUBS:
                for hemi in ["L", "R"]:

                    clean_xyz = stack_surfaces(CLEAN_ROOT, sub, hemi, ses="ses-01")
                    if clean_xyz is None:
                        # If clean is missing, treat as fail (and skip)
                        fail_count += 1
                        continue

                    if i == 0 and j == 0:
                        # baseline: perfect match for each successful clean hemi
                        corrs.append(1.0)
                        continue

                    # pick dataset root (same logic you already have)
                    if i > 0 and j == 0:
                        ds_root = noise_levels[i - 1][1]
                    elif i == 0 and j > 0:
                        ds_root = blur_levels[j - 1][1]
                    else:
                        ds_root = combo_lookup.get((i, j), None)
                        if ds_root is None:
                            fail_count += 1
                            continue

                    degraded_root = ds_root / "derivatives" / "hippunfold"
                    degraded_xyz = stack_surfaces(degraded_root, sub, hemi, ses=None)

                    if degraded_xyz is None:
                        fail_count += 1
                        continue

                    r = pearson_r(clean_xyz, degraded_xyz)
                    if np.isfinite(r):
                        corrs.append(r)
                    else:
                        # Undefined correlation due to zero variance etc.
                        fail_count += 1

            fails[i, j] = fail_count
            nvalid[i, j] = len(corrs)

            if len(corrs) > 0:
                results[i, j] = float(np.mean(corrs))
            else:
                results[i, j] = np.nan

    # ---- plot 3x3 heatmap ----

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(results, cmap="viridis")
    im.cmap.set_bad(color=(0.2, 0.2, 0.2, 1.0))  # dark gray for NA cells

    ax.tick_params(
        axis='both',       # changes apply to both x and y axes
        which='both',       # both major and minor ticks are affected
        bottom=False,       # ticks along the bottom edge are off
        top=False,          # ticks along the top edge are off
        left=False,         # ticks along the left edge are off
        right=False,        # ticks along the right edge are off
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False)

    for i in range(n_rows):
        for j in range(n_cols):
            if np.isfinite(results[i, j]):
                txt = f"{results[i, j]:.4f}\n(n={nvalid[i,j]}, fail={fails[i,j]})"
            else:
                txt = f"NA\n(n={nvalid[i,j]}, fail={fails[i,j]})"
            ax.text(j, i, txt, ha="center", va="center", color="red", fontsize=8)

    ax.set_title("XYZ Coordinate Correlation (vs Clean)")
    plt.tight_layout()

    out_png = Path("plots") / "xyz_correlation_matrix.png"
    out_png.parent.mkdir(exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()