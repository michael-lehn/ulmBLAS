#!/bin/sh
# Profiles getrf2 and potrf2 (profile/factor.cc) on one thread and prints
# the shares of the groups of groups.py.  Usage:
#
#   profile.sh xctrace     macOS, Time Profiler of Xcode (n = 2000)
#   profile.sh perf        Linux, perf (n = 2000)
#   profile.sh callgrind   Linux, valgrind (instructions, n = 500)
#
#   ULMBLAS: a clone of ulmBLAS at tag v0.4.2 (default: ~/git/ulmBLAS)
#   CXX:     the compiler (default: c++)
# On x86-64 with AVX2 ulmBLAS uses an assembly micro kernel, which is
# compiled along (haswell_4x8.S).  For valgrind, which does not know
# AVX-512, the program is compiled for AVX2 only.
cd "$(dirname "$0")"
ULMBLAS=${ULMBLAS:-$HOME/git/ulmBLAS}
CXX=${CXX:-c++}
TOOL=$1
case $(uname -m) in
x86_64)
    ARCH="-march=native"
    [ "$TOOL" = callgrind ] && ARCH="-mavx2 -mfma"
    ASM=$ULMBLAS/ulmblas/level3/ukernel/openblas/asm/haswell_4x8.S ;;
*)
    ARCH="-mcpu=native"
    ASM= ;;
esac
$CXX -std=c++11 -O3 -g -fno-omit-frame-pointer $ARCH -fopenmp-simd \
    -isystem "$ULMBLAS" factor.cc $ASM -o factor || exit 1
export ULMBLAS_NUM_THREADS=1

for w in lu chol; do
    case $TOOL in
    xctrace)
	# Xcode, also if xcode-select points to the Command Line Tools
	export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
	rm -rf $w.trace
	xcrun xctrace record --template 'Time Profiler' --output $w.trace \
	    --target-stdout - --launch -- ./factor $w 2000 100
	xcrun xctrace export --input $w.trace --xpath \
	    '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
	    > $w.xml
	python3 xctrace-top.py $w.xml | python3 groups.py ;;
    perf)
	perf record -q -o $w.data -F 2000 ./factor $w 2000 40
	perf report -i $w.data --stdio --no-children --sort symbol \
	    2>/dev/null | python3 groups.py ;;
    callgrind)
	valgrind --tool=callgrind --callgrind-out-file=$w.cg ./factor $w 500 2
	callgrind_annotate --auto=no $w.cg | python3 groups.py ;;
    *)
	echo "usage: $0 xctrace|perf|callgrind"
	exit 1 ;;
    esac
done
