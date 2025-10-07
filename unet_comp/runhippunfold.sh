#!/usr/bin/env bash
set -euo pipefail

export hippunfold_sif=/data/mica1/01_programs/singularity/hippunfold_dev-v2.0.0.sif
export hippunfold_cache=/host/cassio/export03/data/opt/hippunfold_v2stable/.cache

DATASET=${1:?dataset}
MODEL=${2:?model}

# Prefer per-job scratch from SGE; fallback to /dev/shm
# SCRATCH_ROOT="${TMPDIR:-/dev/shm}"
SCRATCH_ROOT=/export02 # appropriate for bb-comp
SCRATCH="${SCRATCH_ROOT}/hippunfold_${DATASET}_${MODEL}"
SRC_DIR="hippunfold_${DATASET}_${MODEL}"         # network location (cwd must be network)
OUT_DIR="${SRC_DIR}"                              # stage back to same place

# Ensure scratch exists
mkdir -p "${SCRATCH}"

# ---- Caches (host) ----
export SNAKEMAKE_OUTPUT_CACHE="${hippunfold_cache}"
export HIPPUNFOLD_CACHE_DIR="${hippunfold_cache}"
export XDG_CACHE_HOME="${hippunfold_cache}"

# ---- Ensure these are visible INSIDE the container ----
export SINGULARITYENV_SNAKEMAKE_OUTPUT_CACHE="${SNAKEMAKE_OUTPUT_CACHE}"
export SINGULARITYENV_HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR}"
export SINGULARITYENV_XDG_CACHE_HOME="${XDG_CACHE_HOME}"

# (If the cluster uses Apptainer, these work too)
export APPTAINERENV_SNAKEMAKE_OUTPUT_CACHE="${SNAKEMAKE_OUTPUT_CACHE}"
export APPTAINERENV_HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR}"
export APPTAINERENV_XDG_CACHE_HOME="${XDG_CACHE_HOME}"

# ---- Apptainer/Singularity local config & caches (host) ----
WORKCFG="${PWD}/.singularity"
mkdir -p "${WORKCFG}/cache" "${WORKCFG}/tmp"
# Prefer Apptainer names (compatible with current toolchains)
export APPTAINER_CONFIGDIR="${WORKCFG}"
export APPTAINER_CACHEDIR="${WORKCFG}/cache"
export APPTAINER_TMPDIR="${WORKCFG}/tmp"
# Also set Singularity names for compatibility
export SINGULARITY_CONFIGDIR="${WORKCFG}"
export SINGULARITY_CACHEDIR="${WORKCFG}/cache"
export SINGULARITY_TMPDIR="${WORKCFG}/tmp"

# ---- Bind only what you need (paths must exist on node) ----
# Bind the network project root (cwd), the scratch dir, and /data if needed
export APPTAINER_BIND="${PWD}:${PWD},${SCRATCH_ROOT}:${SCRATCH_ROOT},/data:/data,/host:/host"
export SINGULARITY_BIND="${APPTAINER_BIND}"

# Stage-out function runs on any exit
stage_out() {
  rc=$?   # exit code of the script/run
  echo "[INFO] stage_out: script exiting with code $rc"

  # Don't let stage-out failures clobber rc; log but keep going
  if [[ -d "$SCRATCH" ]]; then
    echo "[INFO] Staging results to $OUT_DIR ..."
    # Copy only newer files, preserve perms/links, continue on hiccups
    rsync -aH --partial --update --delete --info=stats2,progress2 "$SCRATCH/" "$OUT_DIR/" || echo "[WARN] rsync stage-out had errors"
    echo "[INFO] Removing scratch $SCRATCH"
    rm -rf "$SCRATCH" || echo "[WARN] failed to remove scratch"
  else
    echo "[WARN] No scratch dir to stage out: $SCRATCH"
  fi

  exit $rc
}
# Fire on normal exit and on common signals
trap stage_out EXIT HUP INT TERM

# ---------- Stage-in ----------
# rsync -aH --partial --info=stats2,progress2 "${SRC_DIR}/" "${SCRATCH}/"

# ---------- Run ----------
singularity run "$hippunfold_sif" "BIDS_${DATASET}" "$SCRATCH" participant \
  --modality T1w \
  --force-nnunet-model "${MODEL}" \
  --filter-T1w space=nativepro \
  --participant-label $(tr '\n' ' ' < "participants-${DATASET}.txt") \
  --cores 32 \
  --rerun-incomplete --keep-going