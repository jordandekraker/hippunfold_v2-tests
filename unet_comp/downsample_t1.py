#!/usr/bin/env python3
"""
Downsample T1w NIfTI to a target voxel size and (optionally) add Gaussian noise.

Usage (single file, backward-compatible):
  python downsample_t1.py --in /path/to/src_T1w.nii.gz \
                          --out /path/to/out_desc-downsampled_T1w.nii.gz \
                          --vox 1.0 1.0 1.0 \
                          --noise-std-frac 0.10

Usage (batch mode over MICs):
  python downsample_t1.py --participants ./participants-MICs.txt \
                          --src-root /data/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0 \
                          --out-root ./BIDS_bMICs \
                          --vox 1.0 1.0 1.0 \
                          --noise-std-frac 0.10

Notes:
- Output directory is created if needed.
- Uses cubic interpolation (order=3).
- Noise is N(0, sigma), where sigma = noise_std_frac * std(resampled_image).
- Saves float32 data with the resampled affine/header.
- Batch mode mirrors the BIDS-ish layout under --out-root.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output


def downsample_with_noise(
    src_path: str,
    out_path: str,
    voxels: Tuple[float, float, float],
    noise_std_frac: float,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Resample `src_path` to `voxels` and add Gaussian noise proportional to std."""
    src_path = str(src_path)
    out_path = str(out_path)
    voxels = tuple(float(v) for v in voxels)
    if len(voxels) != 3:
        raise ValueError("--vox must have exactly 3 values, e.g., 1.0 1.0 1.0")

    img = nib.load(src_path)
    # Cubic interpolation to target isotropic (or anisotropic) resolution
    img_rs = resample_to_output(img, voxel_sizes=voxels, order=3)

    data = img_rs.get_fdata(dtype=np.float32)
    if noise_std_frac and noise_std_frac > 0:
        std = float(np.std(data))
        if std > 0:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.normal(0.0, noise_std_frac * std, size=data.shape).astype(np.float32)
            data = data + noise

    out_img = nib.Nifti1Image(data, img_rs.affine, img_rs.header)
    out_img.update_header()

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_p))


def iter_subjects_from_file(txt_path: Path) -> Iterable[str]:
    """Yield subject IDs from a text file, stripping whitespace and comments.
    Accepts lines like 'sub-XXXX' or 'XXXX' and normalizes to 'sub-XXXX'.
    """
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if not s.startswith("sub-"):
                s = f"sub-{s}"
            yield s


def find_sessions_for_subject(src_root: Path, sub: str) -> Iterable[str]:
    """Yield session names (e.g., 'ses-01') for which the target T1w file exists."""
    # Look for ses-* folders under {src_root}/{sub}/
    sub_dir = src_root / sub
    if not sub_dir.is_dir():
        return
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        ses = ses_dir.name
        anat = ses_dir / "anat"
        t1 = anat / f"{sub}_{ses}_space-nativepro_T1w.nii.gz"
        if t1.is_file():
            yield ses


def batch_process_mics(
    participants_txt: Path,
    src_root: Path,
    out_root: Path,
    voxels: Tuple[float, float, float],
    noise_std_frac: float,
    seed: Optional[int] = 0,
) -> None:
    """Process all subjects/sessions into a mirrored BIDS-like tree under out_root."""
    rng = np.random.default_rng(seed)
    n_found = 0
    n_done = 0
    missing = []

    for sub in iter_subjects_from_file(participants_txt):
        sessions = list(find_sessions_for_subject(src_root, sub))
        if not sessions:
            missing.append((sub, "no sessions with target T1w"))
            continue

        for ses in sessions:
            rel_in = src_root / sub / ses / "anat" / f"{sub}_{ses}_space-nativepro_T1w.nii.gz"
            if not rel_in.is_file():
                missing.append((f"{sub}_{ses}", "missing input T1w"))
                continue

            n_found += 1

            # Mirror BIDS layout under out_root and add desc-bMICs (downsampled+noise)
            out_anat = out_root / sub / ses / "anat"
            out_name = f"{sub}_{ses}_space-nativepro_desc-bMICs_T1w.nii.gz"
            out_path = out_anat / out_name

            try:
                downsample_with_noise(str(rel_in), str(out_path), voxels, noise_std_frac, rng=rng)
                n_done += 1
                print(f"[OK] {sub} {ses} -> {out_path}")
            except Exception as e:
                missing.append((f"{sub}_{ses}", f"failed: {e}"))

    print("\n=== Batch summary ===")
    print(f"Inputs found: {n_found}")
    print(f"Processed:    {n_done}")
    if missing:
        print("Skipped / Issues:")
        for item, why in missing:
            print(f" - {item}: {why}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Downsample a T1w NIfTI and optionally add Gaussian noise (single file or batch over MICs)."
    )
    # Single-file mode (backward-compatible)
    p.add_argument("--in", dest="src", help="Path to source T1w NIfTI (e.g., *_T1w.nii.gz)")
    p.add_argument("--out", dest="out", help="Path to output NIfTI")

    # Batch mode
    p.add_argument(
        "--participants",
        type=str,
        help="Path to participants-MICs.txt (one subject per line; with or without 'sub-' prefix).",
    )
    p.add_argument(
        "--src-root",
        type=str,
        default="/data/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0",
        help="Root containing {sub}/{ses}/anat/{sub}_{ses}_space-nativepro_T1w.nii.gz",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="./BIDS_bMICs",
        help="Output root where the mirrored BIDS-like tree will be written.",
    )

    # Common options
    p.add_argument(
        "--vox",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Target voxel size (mm), three floats, e.g., 1.0 1.0 1.0",
    )
    p.add_argument(
        "--noise-std-frac",
        type=float,
        default=0.10,
        help="Gaussian noise sigma as a fraction of the image std after resampling (default: 0.10)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for noise generation (default: 0).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Decide mode: single-file vs batch
    if args.src and args.out:
        downsample_with_noise(args.src, args.out, args.vox, args.noise_std_frac)
        print(f"[OK] Wrote {args.out}")
        return

    if args.participants:
        participants_txt = Path(args.participants)
        if not participants_txt.is_file():
            raise FileNotFoundError(f"Participants file not found: {participants_txt}")

        src_root = Path(args.src_root)
        if not src_root.is_dir():
            raise NotADirectoryError(f"Source root not found: {src_root}")

        out_root = Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        batch_process_mics(
            participants_txt=participants_txt,
            src_root=src_root,
            out_root=out_root,
            voxels=tuple(args.vox),
            noise_std_frac=float(args.noise_std_frac),
            seed=args.seed,
        )
        return

    raise SystemExit(
        "Please provide either single-file args (--in and --out) "
        "or batch args (--participants [--src-root --out-root])."
    )


if __name__ == "__main__":
    main()
