for d in MICs bMICs PNI; do
  for m in synthseg_v0.2 T1w; do
    echo "qsub -q mica.q -N hippunfold_${d}_${m} -pe smp 32 -l h='bb-comp*' -cwd runhippunfold.sh ${d} ${m}"
    qsub -q mica.q -N hippunfold_${d}_${m} -pe smp 32 -l h='bb-comp*' -cwd runhippunfold.sh ${d} ${m}
  done
done