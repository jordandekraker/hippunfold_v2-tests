#!/usr/bin/env python3
"""
make_noisy_blurry_bids.py

Create multiple BIDS-like output directories with degradation sweeps:
  - noise only (2 levels)
  - blur only (2 levels)
  - blur+noise (all combinations of 2x2 = 4 levels)

Noise is defined as a fraction of the image's robust intensity range:
  std = noise_frac * (P_hi - P_lo), percentiles default to P2 and P98.

Output layout (one root per condition/level):
  <OUT_ROOT>/<tag>/sub-XXXX/anat/sub-XXXX_T1w.nii.gz

Notes
-----
- Output omits 'ses-' and 'desc-' by design.
- For blur+noise, blur is applied BEFORE noise (so blur doesn't "erase" the added noise).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torchio as tio


# =============================================================================
# USER EDIT ZONE
# =============================================================================

# Paste your subject IDs here (no 'sub-' prefix), e.g. ["PNC010", "PNC011", ...]
SUBS: list[str] = [
    "PNC003",
    "PNC007",
    "PNC011",
    "PNC014",
    "PNC019",
    "PNC022",
    "PNC026",
    "PNC020",
    "PNC024",
    "PNC027",
]

# Template path (used only to infer the per-subject path pattern)
TEMPLATE_IN_PATH = Path(
    "/host/bb-comp/tank/data/BIDS_PNI/derivatives/hippunfold_v2.0.0/"
    "sub-PNC010/ses-01/anat/sub-PNC010_ses-01_desc-preproc_T1w.nii.gz"
)

# Output root directory
OUT_ROOT = Path("/host/bb-comp/tank/data/BIDS_PNI_degraded")

# Robust histogram percentiles used to define the intensity range for noise scaling
ROBUST_PCTS: Tuple[float, float] = (2.0, 98.0)

# Two noise levels, expressed as fractions of robust intensity range (P_hi - P_lo)
NOISE_FRACS: list[float] = [0.1, 0.25]  # 10% and 25% of robust range

# Two blur levels, expressed as sigma range in voxel units (TorchIO RandomBlur uses std in voxels)
BLUR_SIGMAS: list[Tuple[float, float]] = [(0.5, 0.8), (1.0, 1.5)]  # (min,max) per level

# Seeds (optional). Set to an int for reproducible randomness across runs.
SEED: int | None = None

# =============================================================================


def build_in_path_for_sub(sub: str) -> Path:
    """
    Build the input path for a subject based on TEMPLATE_IN_PATH.
    Assumes only the 'sub-XXXX' substrings change.
    """
    sub_tag = f"sub-{sub}"
    p = Path(str(TEMPLATE_IN_PATH).replace("sub-PNC010", sub_tag))
    p = Path(
        str(p).replace(
            "sub-PNC010_ses-01_desc-preproc_T1w.nii.gz",
            f"{sub_tag}_ses-01_desc-preproc_T1w.nii.gz",
        )
    )
    return p


def build_out_path(root_dir: Path, sub: str) -> Path:
    """
    Output BIDS-like path omitting ses and desc:
      <root_dir>/sub-XXXX/anat/sub-XXXX_T1w.nii.gz
    """
    sub_tag = f"sub-{sub}"
    out_dir = root_dir / sub_tag / "anat"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{sub_tag}_T1w.nii.gz"


def load_subject(image_path: Path) -> tio.Subject:
    return tio.Subject(T1w=tio.ScalarImage(image_path))


def robust_range_from_subject(subject: tio.Subject, p_lo: float, p_hi: float) -> float:
    """
    Compute robust intensity range from percentiles over finite voxels.
    """
    data = subject.T1w.data  # shape (1, X, Y, Z), torch tensor
    arr = data.detach().cpu().numpy().astype(np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite voxels found in image.")
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    rng = float(hi - lo)
    return max(rng, 1e-6)


def noise_transform_from_frac(subject: tio.Subject, noise_frac: float) -> tio.Transform:
    """
    Create a TorchIO RandomNoise transform with std set as a fraction of robust range.
    """
    p_lo, p_hi = ROBUST_PCTS
    rng = robust_range_from_subject(subject, p_lo, p_hi)
    std = noise_frac * rng
    return tio.RandomNoise(mean=0.0, std=(std, std))


def blur_transform_from_sigma(sigma_min: float, sigma_max: float) -> tio.Transform:
    """
    Create a TorchIO RandomBlur transform (Gaussian blur) with sigma range in voxels.
    """
    return tio.RandomBlur(std=(sigma_min, sigma_max))


def apply_and_save(subject: tio.Subject, transform: tio.Transform | None, out_path: Path) -> None:
    out_subj = transform(subject) if transform is not None else subject
    out_subj.T1w.save(out_path)


def fmt_tag(s: str) -> str:
    # filesystem-friendly: replace '.' with 'p'
    return s.replace(".", "p")


def main(subs: Sequence[str]) -> None:
    if not subs:
        raise SystemExit("SUBS is empty. Paste subject IDs (e.g. 'PNC010') into SUBS at the top.")

    if SEED is not None:
        import torch
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    exp_specs: list[tuple[str, tuple]] = []

    # -------------------------
    # noise only (2 levels)
    # -------------------------
    for i, frac in enumerate(NOISE_FRACS, start=1):
        tag = fmt_tag(f"noise-L{i}_frac{frac:.3f}")
        exp_specs.append((tag, ("noise", frac)))

    # -------------------------
    # blur only (2 levels)
    # -------------------------
    for i, (smin, smax) in enumerate(BLUR_SIGMAS, start=1):
        tag = fmt_tag(f"blur-L{i}_sig{smin:.2f}-{smax:.2f}")
        exp_specs.append((tag, ("blur", smin, smax)))

    # -------------------------
    # blur+noise (ALL combinations: 2 x 2 = 4)
    # blur is applied BEFORE noise
    # -------------------------
    for ni, frac in enumerate(NOISE_FRACS, start=1):
        for bi, (smin, smax) in enumerate(BLUR_SIGMAS, start=1):
            tag = fmt_tag(f"blurnoise-N{ni}B{bi}_frac{frac:.3f}_sig{smin:.2f}-{smax:.2f}")
            exp_specs.append((tag, ("blurnoise", frac, smin, smax)))

    print(f"Writing degraded datasets under: {OUT_ROOT}")
    print(f"Subjects: {len(subs)}")
    print(f"Experiments: {len(exp_specs)}")
    for tag, spec in exp_specs:
        print(f"  {tag}: {spec}")

    for sub in subs:
        in_path = build_in_path_for_sub(sub)
        if not in_path.exists():
            print(f"[WARN] Missing input for {sub}: {in_path}")
            continue

        subj = load_subject(in_path)

        for tag, spec in exp_specs:
            mode = spec[0]
            root_dir = OUT_ROOT / tag

            if mode == "noise":
                frac = spec[1]
                t = noise_transform_from_frac(subj, frac)

            elif mode == "blur":
                smin, smax = spec[1], spec[2]
                t = blur_transform_from_sigma(smin, smax)

            elif mode == "blurnoise":
                frac, smin, smax = spec[1], spec[2], spec[3]
                t_blur = blur_transform_from_sigma(smin, smax)
                t_noise = noise_transform_from_frac(subj, frac)
                # blur -> noise
                t = tio.Compose([t_blur, t_noise])

            else:
                raise RuntimeError(f"Unknown mode: {mode}")

            out_path = build_out_path(root_dir, sub)
            apply_and_save(subj, t, out_path)
            print(f"[OK] {sub} -> {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(SUBS)
