#!/bin/sh
#  build-small.sh B REPO : the programs of this campaign in B.
#
#    B/bench2-published   ulmBLAS 881d321 (published SME2 backend)
#    B/bench2-packtail    d7ca66a (+ packing tail through ZA)
#    B/bench2-cache       2f18d32 (+ per-thread buffers; 9d2b45a only adds a test)
#    B/bench2-direct      d3566f2 (+ direct NEON path)
#    B/bench2-final       b38f3d5 (+ prefetch outside streaming mode)
#
#  each: git archive of the commit, make cblas ARCHFLAGS=-mcpu=apple-m4,
#  bench2.c linked with libcblas.a (Apple clang).  bench2-accelerate and
#  bench2-blis-sme, peak and thermal are the programs of the campaign
#  2026-10-25-macbook-m5-sme-square (see its build scripts; checksums in
#  data/programs.sha256).
set -u
B=${1:?usage: build-small.sh B REPO}; REPO=${2:?usage: build-small.sh B REPO}
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p $B/lib || exit 1; : > $B/build.log
for vc in published:881d321 packtail:d7ca66a cache:2f18d32 direct:d3566f2 \
          final:b38f3d5; do
    v=${vc%%:*}; c=${vc#*:}; L=$B/lib/$v
    rm -rf $L; mkdir -p $L
    (cd $REPO && git archive $c) | tar -x -C $L || exit 1
    echo "$v: ulmBLAS $(cd $REPO && git rev-parse $c)" >> $B/build.log
    (cd $L && make -j8 cblas ARCHFLAGS=-mcpu=apple-m4) >> $B/build.log 2>&1 \
        || { echo "build $v failed"; exit 1; }
    /usr/bin/clang -O2 -I$L $S/bench2.c $L/libcblas.a -lc++ -lm \
        -o $B/bench2-$v || exit 1
done
echo "built: $(cd $B && /bin/ls bench2-* | tr '\n' ' ')"
