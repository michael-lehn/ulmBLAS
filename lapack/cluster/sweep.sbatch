#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=sweep-%j.out
#
#  sweep.sbatch SRC OUT ROUTINE ROUND ["INCS"] ["SIZES"] ["VARS"]: row
#  stride sweep.  For every row stride R (--inc-row, column stride R*n) and
#  variant (default: inplace, colmajor; tiled for potrf) the sizes SIZES,
#  one process per (R, variant, n).  The order of the strides changes from
#  round to round.  A variant is VARIANT[+pp][+oS[+wW][+strip|+lean]]:
#  --variant VARIANT, +pp: --prepack 1, +oS: --outer S, +wW: --window W,
#  +strip: --strip 1, +lean: --strip 2.
#  Raw output: OUT/raw/sweep-rROUND-ROUTINE-incR-VARIANT.txt.
#
set -euo pipefail
SRC=$1; OUT=$2; R=$3; RND=$4
INCS=${5:-"1 2 3 4 6 8 12 16 24 32 42 64 96 128"}
SIZES=${6:-"500 1000 2000 4000"}
source $SRC/lapack/cluster/env.sh
cd $SRC/lapack
[ -x build/bench-ulmblas ] || bash cluster/build.sh $SRC
mkdir -p $OUT/raw $OUT/conditions
OPT="--min-time 0.3 --min-samples 7 --max-samples 300 --warmup 2"
VARS=${7:-}
if [ -z "$VARS" ]; then
    VARS="inplace colmajor"; [ "$R" = potrf ] && VARS="inplace colmajor tiled"
fi
cond() {
    { echo "$(date -Is) $1"; uptime; grep MHz /proc/cpuinfo | awk '{s+=$4; n++} END {printf "mean MHz %.0f over %d cpus\n", s/n, n}'; } \
        >> $OUT/conditions/$SLURM_JOB_ID.txt
}
{ echo "host $(hostname) job $SLURM_JOB_ID"; lscpu; env | grep -E '^OMP_|^OPENBLAS' | sort
  echo "numactl: ${NUMA[*]}"; g++ --version | head -1
  echo "transparent hugepages: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
  free -g | head -2; } > $OUT/conditions/$SLURM_JOB_ID-env.txt
LIST=($INCS); nl=${#LIST[@]}
for ((v=0; v<nl; ++v)); do
    inc=${LIST[$(( (v+RND) % nl ))]}
    cond "round $RND $R inc $inc start"
    for vari in $VARS; do
        out=$OUT/raw/sweep-r$RND-$R-inc$inc-$vari.txt
        : > $out
        for n in $SIZES; do
            base=${vari%%+*}; xopt=""
            case "$vari" in *+pp*) xopt="$xopt --prepack 1";; esac
            so=$(echo "$vari" | sed -n 's/.*+o\([0-9]*\).*/\1/p')
            [ -n "$so" ] && xopt="$xopt --outer $so"
            sw=$(echo "$vari" | sed -n 's/.*+w\([0-9]*\).*/\1/p')
            [ -n "$sw" ] && xopt="$xopt --window $sw"
            case "$vari" in *+strip*) xopt="$xopt --strip 1";; esac
            case "$vari" in *+lean*) xopt="$xopt --strip 2";; esac
            "${NUMA[@]}" ./build/bench-ulmblas $R $n $n 1 $OPT --inc-row $inc \
                --variant $base $xopt >> $out
        done
    done
done
cond end
