export PYTHONNOUSERSITE=1

python ../hippunfold/hippunfold/run.py /data/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0 BIDS_PNI/hippunfold_v2.0.0_synthlayer participant --participant_label $(cat BIDS_PNI/participants.txt) --filter-T1w space=nativepro session=01 --modality T1w --force_nnunet_model synthlayer_v0.3 --inject_template layers --cores all --notemp --keep-going --rerun-incomplete

python ../hippunfold/hippunfold/run.py /data/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0 BIDS_PNI/hippunfold_v2.0.0_synthlayer participant --participant_label $(cat BIDS_PNI/participants.txt) --filter-T1w space=nativepro session=02 --modality T1w --force_nnunet_model synthlayer_v0.3 --inject_template layers --cores all --notemp --keep-going --rerun-incomplete

python ../hippunfold/hippunfold/run.py /data/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0 BIDS_PNI/hippunfold_v2.0.0_synthlayer participant --participant_label $(cat BIDS_PNI/participants.txt) --filter-T1w space=nativepro session=03 --modality T1w --force_nnunet_model synthlayer_v0.3 --inject_template layers --cores all --notemp --keep-going --rerun-incomplete

python ../hippunfold/hippunfold/run.py /data/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0 BIDS_MICs/hippunfold_v2.0.0_synthlayer participant --participant_label $(cat BIDS_MICs/participants.txt) --filter-T1w space=nativepro session=01 --modality T1w --force_nnunet_model synthlayer_v0.3 --inject_template layers --cores all --notemp --keep-going --rerun-incomplete

python ../hippunfold/hippunfold/run.py /data/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0 BIDS_MICs/hippunfold_v2.0.0_synthlayer participant --participant_label $(cat BIDS_MICs/participants.txt) --filter-T1w space=nativepro session=02 --modality T1w --force_nnunet_model synthlayer_v0.3 --inject_template layers --cores all --notemp --keep-going --rerun-incomplete
