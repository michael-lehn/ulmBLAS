#!/bin/sh
#  build-blocks.sh B REPO COMMIT MC_KC_NC... : for each MC_KC_NC a copy of
#  the ulmBLAS tree of COMMIT with the block sizes for double of the SME2
#  configuration replaced (ulmblas/config/blocksize.h), built with
#  ARCHFLAGS=-mcpu=apple-m4 (make cblas), and bench2.c linked against it:
#  B/bench2-bMC_KC_NC.  Build output in B/lib/bMC_KC_NC/build.log.
set -u
B=${1:?}; REPO=${2:?}; C=${3:?}; shift 3
S=$(cd "$(dirname "$0")" && pwd)
for cfg in "$@"; do
    set -- $(echo $cfg | tr _ ' ')
    L=$B/lib/b$cfg; rm -rf $L; mkdir -p $L
    (cd $REPO && git archive $C) | tar -x -C $L || exit 1
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
    (cd $L && make -j8 cblas ARCHFLAGS=-mcpu=apple-m4 > build.log 2>&1) || { echo "build $cfg failed"; exit 1; }
    /usr/bin/clang -O2 -I$L $S/bench2.c $L/libcblas.a -lc++ -lm -o $B/bench2-b$cfg || exit 1
    echo "b$cfg: $(awk '/defined\(USE_SME\)/{f=1} f&&/double, T/{printf "%s ", $NF} f&&/^#elif/&&!/USE_SME/{exit}' $L/ulmblas/config/blocksize.h)"
done
