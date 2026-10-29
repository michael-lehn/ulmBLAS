#!/bin/sh
#  build-config.sh B REPO COMMIT MC_KC_NC... : for each MC_KC_NC a fresh copy
#  of the ulmBLAS tree of COMMIT (git archive) in which only the block sizes
#  for double of the SME2 configuration (ulmblas/config/blocksize.h) are
#  replaced, built with ARCHFLAGS=-mcpu=apple-m4 (make cblas, Apple clang),
#  and the programs linked against it:
#
#    B/bench2-bMC_KC_NC   bench2.c (timing)
#    B/verify-bMC_KC_NC   verify-full.c (all elements of C checked)
#
#  Checks that the build really uses the requested block sizes: blocks.cc,
#  compiled with the flags of the library build against the copied tree,
#  prints BlockSize<double> and BlockSizeUGemm<double> as the library sees
#  them (USE_SME must be defined); a mismatch stops the script.  Per
#  configuration B/lib/bMC_KC_NC/config.txt records the commit, the block
#  sizes, the compile command and the SHA-256 of the library and programs;
#  build output in B/lib/bMC_KC_NC/build.log.
set -u
B=${1:?}; REPO=${2:?}; C=${3:?}; shift 3
S=$(cd "$(dirname "$0")" && pwd)
CC=/usr/bin/clang; CXX=/usr/bin/clang++
commit=$(cd $REPO && git rev-parse $C) || exit 1
for cfg in "$@"; do
    set -- $(echo $cfg | tr _ ' ')
    L=$B/lib/b$cfg
    rm -rf "${L:?}"; mkdir -p $L
    (cd $REPO && git archive $commit) | tar -x -C $L || exit 1
    python3 -I - $L/ulmblas/config/blocksize.h $1 $2 $3 <<'PY' || exit 1
import sys
p, mc, kc, nc = sys.argv[1:]
s = open(p).read()
i = s.index('#elif defined(USE_SME)'); j = s.index('#elif', i + 5)
seg = s[i:j]
for name, val in (('MC', mc), ('KC', kc), ('NC', nc)):
    k = seg.index('static constexpr int %s' % name)
    a = seg.index('std::is_same<double, T>::value               ? ', k)
    e = seg.index('\n', a)
    seg = seg[:a] + 'std::is_same<double, T>::value               ? ' + val + seg[e:]
open(p, 'w').write(s[:i] + seg + s[j:])
PY
    (cd $L && make -j8 cblas CXX=$CXX ARCHFLAGS=-mcpu=apple-m4 > build.log 2>&1) \
        || { echo "build $cfg failed"; exit 1; }
    $CXX -std=c++11 -mcpu=apple-m4 -I$L $S/blocks.cc -o $L/blocks || exit 1
    got=$($L/blocks)
    if [ "$got" != "USE_SME MC $1 KC $2 NC $3 MR 32 NR 16" ]; then
        echo "b$cfg: wrong configuration: $got"; exit 1
    fi
    $CC -O2 -I$L $S/bench2.c $L/libcblas.a -lc++ -lm -o $B/bench2-b$cfg || exit 1
    $CC -O2 -I$L $S/verify-full.c $L/libcblas.a -lc++ -lm -o $B/verify-b$cfg || exit 1
    {
        echo "config b$cfg"
        echo "ulmBLAS $commit"
        echo "blocks $got"
        echo "compile $(cat $L/interfaces/blas/C/build.cblas/compile.cmd 2>/dev/null)"
        echo "compiler $($CXX --version | head -1)"
        (cd $B && shasum -a 256 lib/b$cfg/libcblas.a bench2-b$cfg verify-b$cfg)
    } > $L/config.txt
    echo "b$cfg: $got"
done
