#!/bin/sh
#  build-lib.sh BASE NAME SRC : build the CBLAS library of the ulmBLAS
#  source tree SRC (copied to BASE/lib/NAME, make cblas, Apple clang) and
#  the benchmark programs BASE/bin/bench2-NAME (dgemm) and bench3-NAME
#  (dsyrk, dsymm, dtrmm, dtrsm) of the M5 campaign 2026-10-24.
set -u
BASE=${1:?}; NAME=${2:?}; SRC=${3:?}; S=$(cd "$(dirname "$0")" && pwd)
L=$BASE/lib/$NAME
rm -rf $L; mkdir -p $BASE/lib $BASE/bin; cp -R $SRC $L || exit 1
(cd $L && make -j8 cblas) > $BASE/lib/build-$NAME.log 2>&1 || { echo "build $NAME failed"; exit 1; }
for p in bench2 bench3; do
    /usr/bin/clang -O2 -I$L $S/$p.c $L/libcblas.a -lc++ -lm -o $BASE/bin/$p-$NAME || exit 1
done
grep -m1 -E 'gemm\.cc' $BASE/lib/build-$NAME.log | sed -E 's/ -o .*//; s/  +/ /g'
