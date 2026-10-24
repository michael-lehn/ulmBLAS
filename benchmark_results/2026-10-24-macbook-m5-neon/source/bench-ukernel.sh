#!/bin/sh
#  bench-ukernel.sh DIR : throughput of the micro kernels alone (ukernel.cc,
#  built with Apple clang from DIR/src-neon: NEON and, with
#  -DULMBLAS_NO_NEON, the reference kernel), with the thermal guard
#  (guard.sh).  Results in DIR/bench/res-ukernel.
set -u
D=${1:?}; B=$D/bench; LOG=$B/run-ukernel
S=$(cd "$(dirname "$0")" && pwd); . $S/guard.sh
mkdir -p $B/res-ukernel; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
F="-std=c++11 -O3 -fomit-frame-pointer -I$D/src-neon"
c++ $F $S/ukernel.cc -o ukernel-neon && c++ $F -DULMBLAS_NO_NEON $S/ukernel.cc -o ukernel-ref || exit 1
echo "== ukernel start $(date '+%F %T'), $(status)"
measure res-ukernel/neon.txt ./ukernel-neon 4 8 16 32 64 128 256 512 1024 2048 3072
measure res-ukernel/ref.txt  ./ukernel-ref  4 8 16 32 64 128 256 512 1024 2048 3072
echo "== ukernel done $(date '+%T'), $(status)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
