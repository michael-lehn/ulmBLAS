#!/bin/sh
#  build-variants.sh DIR VARIANT... : build the benchmark programs of the
#  variants below DIR/bench (macOS, Apple clang).  Source trees: DIR/src-base
#  (the ulmBLAS commit before the NEON kernel) and DIR/src-neon (the NEON
#  kernel).  Variants:
#      base           src-base, default build (reference kernels, 4 x 4)
#      ref            src-neon, CPPFLAGS=-DULMBLAS_NO_NEON (reference kernels)
#      neon           src-neon, default build (block sizes of blocksize.h)
#      neon-MC-KC     src-neon, MC and KC for double set by setblocks.py
#      neon-gcc       src-neon, default build with CXX=g++-16
#      blis openblas  DIR/refs (build-refs.sh)
#      accelerate     Apple Accelerate (CBLAS of the system, for context)
set -u
D=${1:?}; shift; B=$D/bench; R=$D/refs; S=$(cd "$(dirname "$0")" && pwd)
CC=/usr/bin/clang; CFLAGS=-O2
mkdir -p $B; cd $B || exit 1
fail() { echo "FAILED (exit $1): $2" | tee -a $B/failures; }
link() {   # NAME INCLUDE LIBS...
    n=$1; inc=$2; shift 2
    $CC $CFLAGS -I$inc $S/bench2.c "$@" -o bench-$n || fail $? "link bench2 $n"
    $CC $CFLAGS -I$inc $S/bench3.c "$@" -o bench3-$n || fail $? "link bench3 $n"
}
for v in "$@"; do
    case $v in
    base|ref|neon|neon-gcc|neon-*-*)
        src=$D/src-neon; mk=""; cpp=""
        case $v in
            base) src=$D/src-base ;;
            ref) cpp=-DULMBLAS_NO_NEON ;;
            neon-gcc) mk="CXX=g++-16 CC=gcc-16" ;;
        esac
        rm -rf v-$v; cp -R $src v-$v
        case $v in
            neon-gcc) ;;
            neon-*-*) python3 $S/setblocks.py v-$v/ulmblas/config/blocksize.h \
                          $(echo $v | cut -d- -f2) $(echo $v | cut -d- -f3) \
                          || { fail $? "block sizes $v"; continue; } ;;
        esac
        (cd v-$v && make -j8 cblas $mk ${cpp:+CPPFLAGS=$cpp}) > build-$v.log 2>&1 || { fail $? "build $v"; continue; }
        if [ $v = neon-gcc ]; then
            #  GCC variant: benchmark programs compiled with gcc-16, linked by
            #  g++-16 (libstdc++ of GCC)
            for p in bench2:bench bench3:bench3; do
                gcc-16 $CFLAGS -Iv-$v -c $S/${p%%:*}.c -o ${p##*:}-$v.o \
                    && g++-16 ${p##*:}-$v.o v-$v/libcblas.a -lm -o ${p##*:}-$v \
                    || fail $? "link ${p%%:*} $v"
                rm -f ${p##*:}-$v.o
            done
        else
            link $v v-$v v-$v/libcblas.a -lc++ -lm
        fi
        echo "$v: $(grep -m1 -E 'gemm\.cc' build-$v.log | sed -E 's/ -o .*//; s/  +/ /g')" ;;
    blis)       link blis $R/blis/include/blis $R/blis/lib/libblis.a -lm -lpthread ;;
    openblas)   link openblas $R/openblas/include $R/openblas/lib/libopenblas.a -lm ;;
    accelerate) link accelerate $S/accelerate -DACCELERATE_NEW_LAPACK -framework Accelerate ;;
    *) fail 2 "unknown variant $v" ;;
    esac
done
