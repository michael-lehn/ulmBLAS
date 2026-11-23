#!/bin/sh
#  build-sweep.sh B REPO : the programs of this campaign in B.
#
#    B/bench2-ulmblas      ulmBLAS v0.4.2 (git archive of the tag), make
#                          cblas ARCHFLAGS=-mcpu=apple-m4 (SME2 backend),
#                          bench2.c linked with libcblas.a
#    B/bench2-accelerate   bench2.c with the Accelerate framework
#                          (accelerate/cblas.h declares its CBLAS)
#    B/bench2-blis-sme     BLIS (BLAS-like Library Instantiation Software)
#                          pull request #921, configuration m4sme_p
#                          (build-blis-sme.sh), linked statically
#    B/peak, B/thermal     NEON peak of the core and the thermal state
#                          (used by guard.sh)
#
#  Apple clang; the commands are written to B/build.log.
set -u
B=${1:?usage: build-sweep.sh B REPO}; REPO=${2:?usage: build-sweep.sh B REPO}
S=$(cd "$(dirname "$0")" && pwd); CC=/usr/bin/clang
mkdir -p $B/lib || exit 1; : > $B/build.log
run() { echo "$*" >> $B/build.log; "$@" >> $B/build.log 2>&1 || { echo "FAILED: $*"; exit 1; }; }
L=$B/lib/ulmblas; rm -rf $L; mkdir -p $L
(cd $REPO && git archive v0.4.2) | tar -x -C $L || exit 1
echo "ulmblas: $(cd $REPO && git rev-parse 'v0.4.2^{commit}')" >> $B/build.log
(cd $L && make -j8 cblas ARCHFLAGS=-mcpu=apple-m4) >> $B/build.log 2>&1 \
    || { echo "build ulmBLAS failed"; exit 1; }
run $CC -O2 -I$L $S/bench2.c $L/libcblas.a -lm -pthread -o $B/bench2-ulmblas
run $CC -O2 -Wno-deprecated-declarations -I$S/accelerate $S/bench2.c \
    -framework Accelerate -o $B/bench2-accelerate
J=8 sh $S/build-blis-sme.sh $B || exit 1
run $CC -O2 -I$B/refs/blis-sme/include/blis $S/bench2.c \
    $B/refs/blis-sme/lib/libblis.a -lm -o $B/bench2-blis-sme
run $CC -O2 $S/peak.c -o $B/peak
run $CC -O2 -framework Foundation $S/thermal.m -o $B/thermal
echo "built: $(cd $B && /bin/ls bench2-* peak thermal | tr '\n' ' ')"
