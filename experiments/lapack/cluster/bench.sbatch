#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=bench-%j.out
#
#  bench.sbatch SRC OUT ROUTINE "ROUNDS" ["INCS"]: final sweep n = 50, 100,
#  ..., 4000 on an exclusive Ice Lake node.  Each (round, variant) is one
#  process over all sizes; the order of the variants changes from round to
#  round.
#
#  Without INCS: the two libraries, column major (as in the campaign
#  2026-11-06); raw output OUT/raw/rROUND-ROUTINE-LIB.txt.
#  With INCS (e.g. "1 2 42 42/colmajor 42/tiled 1/inplace+pp"): ulmBLAS
#  with each row stride (--inc-row) and variant (--variant, default
#  inplace; tiled is skipped for getrf; +pp: --prepack 1), and OpenBLAS
#  (column major only); raw output
#  OUT/raw/rROUND-ROUTINE-LIB-incR[-VARIANT][-pp].txt.  Conditions in
#  OUT/conditions.
#
set -euo pipefail
SRC=$1; OUT=$2; R=$3; ROUNDS=${4:-"1 2 3"}; INCS=${5:-}
source $SRC/experiments/lapack/cluster/env.sh
cd $SRC/experiments/lapack
[ -x build/bench-ulmblas ] || bash cluster/build.sh $SRC
mkdir -p $OUT/raw $OUT/conditions
OPT="--min-time 0.3 --min-samples 7 --max-samples 300 --warmup 2"
cond() {
    { echo "$(date -Is) $1"; uptime; grep MHz /proc/cpuinfo | awk '{s+=$4; n++} END {printf "mean MHz %.0f over %d cpus\n", s/n, n}'
      numactl -H | head -3; } >> $OUT/conditions/$SLURM_JOB_ID.txt
}
{ echo "host $(hostname) job $SLURM_JOB_ID"; lscpu; env | grep -E '^OMP_|^OPENBLAS' | sort
  echo "numactl: ${NUMA[*]}"; g++ --version | head -1; } > $OUT/conditions/$SLURM_JOB_ID-env.txt
if [ -z "$INCS" ]; then
    for rnd in $ROUNDS; do
        if [ $((rnd % 2)) = 1 ]; then LIBS="ulmblas openblas"; else LIBS="openblas ulmblas"; fi
        for lib in $LIBS; do
            cond "round $rnd $R $lib start"
            "${NUMA[@]}" ./build/bench-$lib $R 50 4000 50 $OPT \
                --raw $OUT/raw/samples-r$rnd-$R-$lib.txt \
                > $OUT/raw/r$rnd-$R-$lib.txt
            gzip -f $OUT/raw/samples-r$rnd-$R-$lib.txt
        done
    done
else
    VARIANTS=(openblas:1)
    for tok in $INCS; do
        [ "$R" = getrf ] && [ "${tok#*/}" = tiled ] && continue
        VARIANTS+=(ulmblas:$tok)
    done
    nv=${#VARIANTS[@]}
    for rnd in $ROUNDS; do
        for ((v=0; v<nv; ++v)); do
            var=${VARIANTS[$(( (v+rnd) % nv ))]}
            lib=${var%%:*}; tok=${var#*:}
            inc=${tok%%/*}; vari=inplace
            [ "$tok" != "$inc" ] && vari=${tok#*/}
            pp=0; [ "${vari%+pp}" != "$vari" ] && { pp=1; vari=${vari%+pp}; }
            tag=r$rnd-$R-$lib-inc$inc
            [ $vari != inplace ] || [ $pp = 1 ] && tag=$tag-$vari
            [ $pp = 1 ] && tag=$tag-pp
            VOPT=""; [ $lib = ulmblas ] && VOPT="--variant $vari --prepack $pp"
            cond "round $rnd $R $lib inc $inc $vari start"
            "${NUMA[@]}" ./build/bench-$lib $R 50 4000 50 $OPT --inc-row $inc $VOPT \
                --raw $OUT/raw/samples-$tag.txt > $OUT/raw/$tag.txt
            gzip -f $OUT/raw/samples-$tag.txt
        done
    done
fi
cond end
