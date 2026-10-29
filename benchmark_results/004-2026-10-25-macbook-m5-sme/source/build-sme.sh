#!/bin/sh
#  build-sme.sh B REPO REFS : build the programs of bench-sme.sh in B.
#
#    REPO   the ulmBLAS repository (git): the SME2 variant is built from the
#           commit SME (default feature/sme-dgemm) with ARCHFLAGS=-mcpu=
#           apple-m4, the NEON variant from the commit NEON (default master),
#           both with Apple clang and the default flags of ulmBLAS.
#    REFS   directory with the installed reference libraries:
#           REFS/blis-sme (BLIS pull request #921, m4sme_p), REFS/blis (BLIS
#           2.0, firestorm), REFS/openblas (OpenBLAS 0.3.34).
#
#  The benchmark programs (bench2.c, bench3.c, ukernel.cc, ukernel-blis.c,
#  smepeak.c, thermal.m, peak.c) are in the directory of this script.  The
#  compile commands are written to B/build.log.
set -u
B=${1:?}; REPO=${2:?}; REFS=${3:?}; SME=${SME:-feature/sme-dgemm}; NEON=${NEON:-master}
S=$(cd "$(dirname "$0")" && pwd); CC=/usr/bin/clang; CXX=/usr/bin/clang++
mkdir -p $B/lib || exit 1; : > $B/build.log
run() { echo "$*" >> $B/build.log; "$@" >> $B/build.log 2>&1 || { echo "FAILED: $*"; exit 1; }; }
for v in sme neon; do
    if [ $v = sme ]; then c=$SME; af=-mcpu=apple-m4; else c=$NEON; af=; fi
    L=$B/lib/$v; rm -rf $L; mkdir -p $L
    (cd $REPO && git archive $c) | tar -x -C $L || exit 1
    echo "$v: $(cd $REPO && git rev-parse $c), ARCHFLAGS='$af'" >> $B/build.log
    (cd $L && make -j8 cblas ARCHFLAGS="$af") >> $B/build.log 2>&1 || { echo "build $v failed"; exit 1; }
    for p in bench2 bench3; do
        run $CC -O2 -I$L $S/$p.c $L/libcblas.a -lc++ -lm -o $B/$p-$v
    done
    run $CXX -std=c++11 -O3 $af -I$L $S/ukernel.cc -o $B/ukernel-$v
done
for p in bench2 bench3; do
    run $CC -O2 -I$REFS/blis-sme/include/blis $S/$p.c $REFS/blis-sme/lib/libblis.a -lm -o $B/$p-blis-sme
    run $CC -O2 -I$REFS/blis/include/blis $S/$p.c $REFS/blis/lib/libblis.a -lm -o $B/$p-blis
    run $CC -O2 -I$REFS/openblas/include $S/$p.c $REFS/openblas/lib/libopenblas.a -lm -o $B/$p-openblas
    run $CC -O2 -Wno-deprecated-declarations -I$S/accelerate $S/$p.c -framework Accelerate -o $B/$p-accelerate
done
run $CC -O2 $S/ukernel-blis.c $REFS/blis-sme/lib/libblis.a -lm -o $B/ukernel-blis-sme
run $CC -O2 -mcpu=apple-m4 $S/smepeak.c -o $B/smepeak
run $CC -O2 $S/peak.c -o $B/peak
run $CC -O2 -framework Foundation $S/thermal.m -o $B/thermal
echo "built: $(cd $B && /bin/ls bench2-* bench3-* ukernel-* smepeak peak thermal | tr '\n' ' ')"
