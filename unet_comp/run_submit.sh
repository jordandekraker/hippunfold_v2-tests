conda activate hippunfold-dev
export hippunfold_cache=/host/cassio/export03/data/opt/hippunfold_v2stable/.cache

cd ../hippunfold
git checkout dev-v2.0.0
cd ../unet_comp

for d in MICs PNI; do
  for m in synthseg_v0.2 T1w; do
    python ../hippunfold/hippunfold/run.py BIDS_${d} hippunfold_${d}_${m} participant --modality T1w --filter_T1w space=nativepro --participant_label $(tr '\n' ' ' < "participants-${d}.txt") --force_nnunet_model ${m} --rerun-incomplete --keep-going --cores 32 --scheduler greedy
  done
done

cd ../hippunfold
git checkout synthlayer
cd ../unet_comp

for d in MICs PNI; do
  python ../hippunfold/hippunfold/run.py BIDS_${d} hippunfold_${d}_synthlayer_v0.3 participant --modality T1w --filter_T1w space=nativepro --participant_label $(tr '\n' ' ' < "participants-${d}.txt") --force_nnunet_model synthlayer_v0.3 --rerun-incomplete --keep-going --cores 32 --scheduler greedy
done
