RUN='python ../hippunfold/hippunfold/run.py'
RUNOPTS='--cores 32 --notemp'
TESTCASE='test-hippunfold_v2.0.0'

mkdir $TESTCASE

### fairly standard uses ###

$RUN ../../../hippunfold-wetrun/lowresMRI $TESTCASE/lowresMRI participant --modality T1w $RUNOPTS

$RUN ../../../hippunfold-wetrun/highresMRI $TESTCASE/highresMRI participant --modality T1w --force_nnunet_model synthseg_v0.2 $RUNOPTS

$RUN ../../../hippunfold-wetrun/thickSlice $TESTCASE/thickSlice participant --modality T2w --rigid-reg-template $RUNOPTS

# $RUN ../../../hippunfold-wetrun/histology $TESTCASE/histology participant --modality dsegtissue --hemi L --filter_dsegtissue hemi=L --derivatives histology $RUNOPTS --skip_inject_template_labels --keep-going
# note this will fail for dentate surfaces since the dentate source/sink are not present. However, template shape injection fails badly on this case

$RUN ../../../hippunfold-wetrun/neonate $TESTCASE/neonate participant --modality T1w --template dHCP --force_nnunet_model neonateT1w_v2 $RUNOPTS

$RUN ../../../hippunfold-wetrun/atrophy $TESTCASE/atrophy participant --modality T1w  --force_nnunet_model ADNI_T1w_v1 $RUNOPTS



RUN='singularity run /data/mica1/01_programs/singularity/hippunfold_v1.5.1.sif'
RUNOPTS='--cores 32 --keep-work'
TESTCASE='test-hippunfold_v1.5.1'

mkdir $TESTCASE

### fairly standard uses ###

$RUN ../../../hippunfold-wetrun/lowresMRI $TESTCASE/lowresMRI participant --modality T1w $RUNOPTS

$RUN ../../../hippunfold-wetrun/highresMRI $TESTCASE/highresMRI participant --modality T1w --force_nnunet_model synthseg_v0.2 $RUNOPTS

$RUN ../../../hippunfold-wetrun/thickSlice $TESTCASE/thickSlice participant --modality T2w --rigid-reg-template $RUNOPTS

# $RUN ../../../hippunfold-wetrun/histology $TESTCASE/histology participant --modality dsegtissue --hemi L --filter_dsegtissue hemi=L --derivatives histology $RUNOPTS --skip_inject_template_labels --keep-going
# note this will fail for dentate surfaces since the dentate source/sink are not present. However, template shape injection fails badly on this case

$RUN ../../../hippunfold-wetrun/neonate $TESTCASE/neonate participant --modality T1w --template dHCP --force_nnunet_model neonateT1w_v2 $RUNOPTS

$RUN ../../../hippunfold-wetrun/atrophy $TESTCASE/atrophy participant --modality T1w  --force_nnunet_model ADNI_T1w_v1 $RUNOPTS
