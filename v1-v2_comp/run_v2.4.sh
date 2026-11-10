#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pixi_bin="${PIXI_BIN:-${HOME}/.pixi/bin/pixi}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/export03/data/tmp/hippunfold-v2-tests-pixi/cache}"
export PYTHONNOUSERSITE=1

for dataset in MICs PNI; do
  participants=()
  read -r -a participants <<< "$(tr '\n' ' ' < "${repo_root}/v1-v2_comp/BIDS_${dataset}/participants.txt")"
  sessions=(01 02)
  if [[ "$dataset" == PNI ]]; then
    sessions+=(03)
  fi
  for session in "${sessions[@]}"; do
    "$pixi_bin" run --manifest-path "${repo_root}/pixi.toml" --locked hippunfold \
      "/data/mica3/BIDS_${dataset}/derivatives/micapipe_v0.2.0" \
      "${repo_root}/v1-v2_comp/BIDS_${dataset}/hippunfold_v2.4" participant \
      --participant_label "${participants[@]}" \
      --filter-T1w space=nativepro "session=${session}" \
      --modality T1w --force_tissuepred_model T1w \
      --cores 32 --notemp --keep-going --rerun-incomplete "$@"
  done
done
