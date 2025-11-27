# useful to view training
BASE="PNI_synthlayer_v0.3"
SUB="sub-PNC003"
SES="ses-02"

# BASE="MICs_T1w"
# SUB="sub-HC076"
# SES="ses-01"

mri_convert hippunfold_${BASE}/${SUB}/${SES}/anat/${SUB}_${SES}_desc-preproc_T1w.nii.gz tmp.mgz

freeview -f hippunfold_${BASE}/${SUB}/${SES}/surf/${SUB}_${SES}_hemi-L_space-T1w_den-8k_label-hipp_midthickness.surf.gii:edgecolor=yellow:name=hipp \
-f hippunfold_${BASE}/${SUB}/${SES}/surf/${SUB}_${SES}_hemi-L_space-T1w_den-8k_label-dentate_midthickness.surf.gii:edgecolor=purple:name=dentate \
-f hippunfold_${BASE}/${SUB}/${SES}/surf/${SUB}_${SES}_hemi-R_space-T1w_den-8k_label-hipp_midthickness.surf.gii:edgecolor=yellow:name=hipp \
-f hippunfold_${BASE}/${SUB}/${SES}/surf/${SUB}_${SES}_hemi-R_space-T1w_den-8k_label-dentate_midthickness.surf.gii:edgecolor=purple:name=dentate \
-v tmp.mgz