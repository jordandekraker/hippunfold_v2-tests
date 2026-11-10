#!/usr/bin/env bash
# Invoked by submit_v2.4.py; all paths are recorded in the submission manifest.
set -euo pipefail
run_dir="$1"
mode="$2"
repo_root="$(cd -- "${run_dir}/../../.." && pwd)"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export PIXI_CACHE_DIR=/host/bb-compxg-01/export03/data/tmp/hippunfold-v2-tests-pixi/cache
export HIPPUNFOLD_CACHE_DIR="${repo_root}/.cache/hippunfold-v2.4"
export SNAKEMAKE_CONDA_PREFIX="${HIPPUNFOLD_CACHE_DIR}/conda"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${NSLOTS:-32}"
scratch="${TMPDIR:-/tmp}/hippunfold-${JOB_ID:-manual}-${SGE_TASK_ID:-0}"
mkdir -p "$scratch"
export TMPDIR="$scratch" TMP="$scratch" TEMP="$scratch"
export XDG_CACHE_HOME="${scratch}/cache"
mkdir -p "$XDG_CACHE_HOME"
cd "$repo_root"
run_hu() {
  "${HOME}/.pixi/bin/pixi" run --manifest-path "${repo_root}/pixi.toml" --locked hippunfold "$@"
}
echo "Host: $(hostname -f); job: ${JOB_ID:-manual}; task: ${SGE_TASK_ID:-0}; slots: ${NSLOTS:-32}"

if [[ "$mode" == prepare ]]; then
  # Populate shared downloads and inference environments once, before the array.
  output="${run_dir}/prepare-output"
  common=("${run_dir}/inputs/MICs" "$output")
  options=(--participant_label HC001 --filter-T1w space=nativepro session=01
    --modality T1w --force_tissuepred_model T1w --cores "${NSLOTS:-4}"
    --notemp --rerun-incomplete)
  run_hu "${common[@]}" download "${options[@]}"
  run_hu "${common[@]}" participant "${options[@]}" --conda-create-envs-only
  touch "${run_dir}/prepare.complete"
  echo 'Shared model, template, atlas, and inference environment preparation completed.'
  exit 0
fi

[[ "$mode" == batch ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
[[ -f "${run_dir}/prepare.complete" ]] || { echo 'Preparation failed; see prepare job log.' >&2; exit 1; }
row="$(awk -F '\t' -v task="${SGE_TASK_ID:?Missing SGE_TASK_ID}" '$1 == task {print}' "${run_dir}/batches.tsv")"
[[ -n "$row" ]] || { echo 'No batch found for this task.' >&2; exit 1; }
IFS=$'\t' read -r task dataset subject_text <<< "$row"
read -r -a subjects <<< "$subject_text"
output="${repo_root}/v1-v2_comp/BIDS_${dataset}/hippunfold_v2.4/.batches/batch-${task}"
mkdir -p "$(dirname -- "$output")"
sessions=(01 02)
[[ "$dataset" != PNI ]] || sessions+=(03)
echo "Dataset: $dataset; subjects: ${subjects[*]}; sessions: ${sessions[*]}; output: $output"
status=0
for session in "${sessions[@]}"; do
  run_hu "${run_dir}/inputs/${dataset}" "$output" participant \
    --participant_label "${subjects[@]}" --filter-T1w space=nativepro "session=${session}" \
    --modality T1w --force_tissuepred_model T1w --cores "${NSLOTS:-32}" \
    --notemp --keep-going --rerun-incomplete || status=1
done
[[ "$status" == 0 ]] || { echo 'At least one session failed; inspect this task log.' >&2; exit "$status"; }
# Each task owns distinct subjects. Publish links without merging Snakemake state.
for subject in "${subjects[@]}"; do
  source_dir="${output}/sub-${subject}"
  destination="${repo_root}/v1-v2_comp/BIDS_${dataset}/hippunfold_v2.4/sub-${subject}"
  [[ -d "$source_dir" ]] || { echo "Missing result: $source_dir" >&2; exit 1; }
  target=".batches/batch-${task}/sub-${subject}"
  if [[ -L "$destination" && "$(readlink "$destination")" == "$target" ]]; then
    continue
  fi
  [[ ! -e "$destination" && ! -L "$destination" ]] || { echo "Refusing to replace $destination" >&2; exit 1; }
  ln -s "$target" "$destination"
done
touch "${run_dir}/task-${task}.complete"
echo "Batch ${task} completed."
