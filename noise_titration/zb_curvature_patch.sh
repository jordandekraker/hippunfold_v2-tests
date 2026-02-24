#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# User settings
# -------------------------
OUT_ROOT="/host/verges/tank/data/BIDS_MICs/derivatives/zbrains_2.0_hp1"
HIPPUNFOLD_ROOT="/data/mica3/BIDS_MICs/derivatives/hippunfold_v1.3.0/hippunfold"

DEN="0p5mm"
LABEL="hipp"
SURF="midthickness"
FEATURE="curvature"
SMOOTH_MM="5"

# If wb_command isn't on PATH, set it explicitly:
WB_COMMAND="${WB_COMMAND:-wb_command}"

# -------------------------
# Main
# -------------------------
shopt -s nullglob

for subdir in "${OUT_ROOT}"/sub-*; do
  [[ -d "${subdir}" ]] || continue
  sub="$(basename "${subdir}")"

  for sesdir in "${subdir}"/ses-*; do
    [[ -d "${sesdir}" ]] || continue
    ses="$(basename "${sesdir}")"

    out_map_dir="${sesdir}/maps/hippocampus"
    mkdir -p "${out_map_dir}"

    for hemi in L R; do
      in_fn="${HIPPUNFOLD_ROOT}/${sub}/${ses}/surf/${sub}_${ses}_hemi-${hemi}_space-T1w_den-${DEN}_label-hipp_${FEATURE}.shape.gii"
      in_surf="${HIPPUNFOLD_ROOT}/${sub}/${ses}/surf/${sub}_${ses}_hemi-${hemi}_space-T1w_den-${DEN}_label-hipp_${SURF}.surf.gii"

      out_fn="${out_map_dir}/${sub}_${ses}_hemi-${hemi}_den-${DEN}_label-hipp_${SURF}_feature-${FEATURE}_smooth-${SMOOTH_MM}mm.func.gii"

      if [[ ! -f "${in_fn}" ]]; then
        echo "[skip] missing metric: ${in_fn}"
        continue
      fi
      if [[ ! -f "${in_surf}" ]]; then
        echo "[skip] missing surf:   ${in_surf}"
        continue
      fi

      tmp_fn="$(mktemp --suffix=".func.gii")"
      trap 'rm -f "${tmp_fn}"' EXIT

      if [[ "${hemi}" == "R" ]]; then
        # flip sign for right hemi only
        "${WB_COMMAND}" -metric-math -x "${tmp_fn}" -var x "${in_fn}"
      else
        # left hemi: no flip, just copy into tmp
        cp -f "${in_fn}" "${tmp_fn}"
      fi

      echo "[run] ${sub} ${ses} hemi-${hemi} -> $(basename "${out_fn}")"
      "${WB_COMMAND}" -metric-smoothing "${in_surf}" "${tmp_fn}" "${SMOOTH_MM}" "${out_fn}"

      rm -f "${tmp_fn}"
      trap - EXIT
    done
  done
done

echo "Done."
