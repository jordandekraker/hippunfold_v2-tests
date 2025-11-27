RUN='python ../hippunfold/hippunfold/run.py'
RUNOPTS='--cores 32 --notemp'
TESTCASE='test-hippunfold_v2.0.0'

mkdir $TESTCASE

$RUN ../../../hippunfold-wetrun/lowresMRI $TESTCASE/lowresMRI participant --modality T1w $RUNOPTS
$RUN ../../../hippunfold-wetrun/highresMRI $TESTCASE/highresMRI participant --modality T1w --force_nnunet_model synthseg_v0.2 $RUNOPTS
$RUN ../../../hippunfold-wetrun/thickSlice $TESTCASE/thickSlice participant --modality T2w --rigid-reg-template $RUNOPTS
$RUN ../../../hippunfold-wetrun/neonate $TESTCASE/neonate participant --modality T1w --template dHCP --force_nnunet_model neonateT1w_v2 $RUNOPTS
$RUN ../../../hippunfold-wetrun/atrophy $TESTCASE/atrophy participant --modality T1w --force_nnunet_model ADNI_T1w_v1 $RUNOPTS
$RUN ../../../hippunfold-wetrun/mouse $TESTCASE/mouse participant --modality T1w  --use_template_seg --template ABAv3 --no-unfolded-reg --crop_res 0.02x0.02x0.02mm $RUNOPTS
$RUN ../../../hippunfold-wetrun/marmoset $TESTCASE/marmoset participant --modality T1w  --use_template_seg --template MBMv3 --no-unfolded-reg --crop_res 0.1x0.1x0.1mm $RUNOPTS
# checklut synthlayer branch
$RUN ../../../hippunfold-wetrun/highresMRI $TESTCASE/synthlayer participant --modality T1w --force_nnunet_model synthlayer_v0.3 --inject_template layers $RUNOPTS


### now let's run only things that were availabel in v1.0

RUN='singularity run /data/mica1/01_programs/singularity/hippunfold_v1.5.1.sif'
RUNOPTS='--cores 32 --keep-work'
TESTCASE='test-hippunfold_v1.5.1'

mkdir $TESTCASE

### fairly standard uses ###

$RUN ../../../hippunfold-wetrun/lowresMRI $TESTCASE/lowresMRI participant --modality T1w $RUNOPTS
$RUN ../../../hippunfold-wetrun/highresMRI $TESTCASE/highresMRI participant --modality T1w $RUNOPTS
$RUN ../../../hippunfold-wetrun/thickSlice $TESTCASE/thickSlice participant --modality T2w --rigid-reg-template $RUNOPTS
