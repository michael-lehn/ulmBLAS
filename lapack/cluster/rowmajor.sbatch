#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=rowmajor-%j.out
#
#  rowmajor.sbatch SRC OUT ROUTINE "ROUNDS": n = 50, 100, ..., 4000, row
#  major vs. column major.  Per round one process for each of
#
#      openblas colmajor    dgetrf_ / dpotrf_
#      openblas rowmajor    LAPACKE, LAPACK_ROW_MAJOR (copy, factor, copy)
#      ulmblas  rowmajor    in place, incRow = n, incCol = 1
#      ulmblas  colmajor    in place (reference)
#
#  in an order that changes from round to round.  Raw output
#  OUT/raw/rROUND-ROUTINE-LIB-LAYOUT.txt, conditions in OUT/conditions.
#
set -euo pipefail
SRC=$1; OUT=$2; R=$3; ROUNDS=${4:-"1 2 3"}
source $SRC/lapack/cluster/env.sh
cd $SRC/lapack
[ -x build/bench-ulmblas ] || bash cluster/build.sh $SRC
mkdir -p $OUT/raw $OUT/conditions
OPT="--min-time 0.3 --min-samples 7 --max-samples 300 --warmup 2"
cond() {
    { echo "$(date -Is) $1"; uptime; grep MHz /proc/cpuinfo | awk '{s+=$4; n++} END {printf "mean MHz %.0f over %d cpus\n", s/n, n}'; } \
        >> $OUT/conditions/$SLURM_JOB_ID.txt
}
{ echo "host $(hostname) job $SLURM_JOB_ID"; lscpu; env | grep -E '^OMP_|^OPENBLAS' | sort
  echo "numactl: ${NUMA[*]}"; g++ --version | head -1; } > $OUT/conditions/$SLURM_JOB_ID-env.txt
VARIANTS=(openblas:colmajor openblas:rowmajor ulmblas:rowmajor ulmblas:colmajor)
nv=${#VARIANTS[@]}
for rnd in $ROUNDS; do
    for ((v=0; v<nv; ++v)); do
        var=${VARIANTS[$(( (v+rnd) % nv ))]}
        lib=${var%%:*}; lay=${var#*:}
        cond "round $rnd $R $lib $lay start"
        "${NUMA[@]}" ./build/bench-$lib $R 50 4000 50 $OPT --layout $lay \
            > $OUT/raw/r$rnd-$R-$lib-$lay.txt
    done
done
cond end
