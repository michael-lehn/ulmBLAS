#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=core-%j.out
#
#  core.sbatch SRC OUT PREC ROUTINE "ROUNDS": ?getrf or ?potrf for one
#  precision (s, d, c, z), n = 50, 100, ..., 4000, one socket, 32 threads:
#
#      ulmblas    the templates of ulmblas/lapack (OpenMP tasks, libgomp)
#      openblas   OpenBLAS 0.3.34 (OpenMP build)
#      mkl        oneMKL 2025.0 (mkl_rt, Intel threading, as MKL.jl)
#
#  One process per (round, library), the order of the libraries changes
#  from round to round.  Raw output OUT/raw/rROUND-PREC-ROUTINE-LIB.txt,
#  conditions in OUT/conditions.
#
set -euo pipefail
SRC=$1; OUT=$2; P=$3; R=$4; ROUNDS=${5:-"1 2 3"}
source $SRC/experiments/lapack/cluster/env.sh
module load numlib/mkl/2025.0 >/dev/null 2>&1
IOMP=/software/bwhpc/common/toolkit/Intel_OneAPI/compiler/2025.0/lib
cd $SRC/experiments/lapack
[ -x build/bench-core-ulmblas ] || bash cluster/build-core.sh $SRC
mkdir -p $OUT/raw $OUT/conditions
OPT="--min-time 0.3 --min-samples 7 --max-samples 300 --warmup 2"
cond() {
    { echo "$(date -Is) $1"; uptime; grep MHz /proc/cpuinfo | awk '{s+=$4; n++} END {printf "mean MHz %.0f over %d cpus\n", s/n, n}'; } \
        >> $OUT/conditions/$SLURM_JOB_ID.txt
}
{ echo "host $(hostname) job $SLURM_JOB_ID"; lscpu; env | grep -E '^OMP_|^OPENBLAS|^MKL' | sort
  echo "numactl: ${NUMA[*]}"; g++ --version | head -1; echo "MKLROOT $MKLROOT"; } \
    > $OUT/conditions/$SLURM_JOB_ID-env.txt
LIBS=(ulmblas openblas mkl)
for rnd in $ROUNDS; do
    for ((v=0; v<3; ++v)); do
        lib=${LIBS[$(( (v+rnd) % 3 ))]}
        cond "round $rnd $P$R $lib start"
        if [ $lib = mkl ]; then
            LD_LIBRARY_PATH=$IOMP:$LD_LIBRARY_PATH MKL_NUM_THREADS=$THREADS \
            MKL_THREADING_LAYER=INTEL "${NUMA[@]}" ./build/bench-core-mkl \
                $P $R 50 4000 50 $OPT > $OUT/raw/r$rnd-$P-$R-$lib.txt
        else
            "${NUMA[@]}" ./build/bench-core-$lib $P $R 50 4000 50 $OPT \
                > $OUT/raw/r$rnd-$P-$R-$lib.txt
        fi
    done
done
cond end
