#!/usr/bin/env python3
"""
run_hippunfold_on_degraded.py

For each degraded BIDS dataset inside BIDS_PNI_degraded,
create a derivatives folder and run:

    hippunfold <indir> <outdir> participant --modality T1w --cores all

Assumes each degradation level is its own BIDS root like:

BIDS_PNI_degraded/
    noise-L1_frac0p020/
        sub-XXXX/anat/sub-XXXX_T1w.nii.gz
    blur-L1_sig0p50-0p80/
        sub-XXXX/anat/sub-XXXX_T1w.nii.gz
    ...

Outputs:
    <degraded_root>/derivatives/hippunfold/
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys


# =============================================================================
# USER EDIT ZONE
# =============================================================================

DEGRADED_ROOT = Path("/host/bb-comp/tank/data/BIDS_PNI_degraded")

# Set to False if you want to see the commands but not execute them
RUN = True

# If True, skip datasets that already contain derivatives/hippunfold
SKIP_IF_EXISTS = True

# =============================================================================


def is_bids_dataset(path: Path) -> bool:
    """
    Minimal check: contains at least one sub-*/anat/*.nii.gz
    """
    return any(path.glob("sub-*/anat/*.nii.gz"))


def main():

    if not DEGRADED_ROOT.exists():
        sys.exit(f"Degraded root not found: {DEGRADED_ROOT}")

    dataset_dirs = [
        p for p in DEGRADED_ROOT.iterdir()
        if p.is_dir() and p.name != "_qc"
    ]

    if not dataset_dirs:
        sys.exit("No degradation datasets found.")

    print(f"\nFound {len(dataset_dirs)} degradation datasets:\n")

    for ds in sorted(dataset_dirs):

        outdir = ds / "derivatives" / "hippunfold"
        outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "hippunfold",
            str(ds),
            str(outdir),
            "participant",
            "--modality", "T1w",
            "--cores", "all",
            "--keep-going", "--rerun-incomplete"
        ]

        print(f"[RUN] {ds.name}")
        print("      ", " ".join(cmd))

        if RUN:
            result = subprocess.run(cmd)

    print("\nDone.")


if __name__ == "__main__":
    main()
