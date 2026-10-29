#!/bin/sh
#  build.sh DIR SRC NAME : build the profiling programs DIR/bin/prof-NAME
#  (ulmBLAS source tree SRC) and, with NAME = master, prof-blis and
#  prof-openblas (BLIS and OpenBLAS of DIR/refs) and thermal, peak.
set -u
D=${1:?}; SRC=${2:?}; NAME=${3:?}; S=$(cd "$(dirname "$0")" && pwd)
R=${REFS:-$D/refs}; CXX=${CXX:-c++}
F="-std=c++14 -O3 -fomit-frame-pointer -I$SRC"
mkdir -p $D/bin
$CXX $F $S/prof.cc -o $D/bin/prof-$NAME || exit 1
if [ $NAME = master ]; then
    $CXX $F -DWITH_CBLAS -I$R/blis/include $S/prof.cc $R/blis/lib/libblis.a -o $D/bin/prof-blis || exit 1
    $CXX $F -DWITH_CBLAS -I$R/openblas/include $S/prof.cc $R/openblas/lib/libopenblas.a -o $D/bin/prof-openblas || exit 1
    cc -O2 -framework Foundation $S/thermal.m -o $D/bin/thermal || exit 1
    cc -O2 $S/peak.c -o $D/bin/peak || exit 1
fi
