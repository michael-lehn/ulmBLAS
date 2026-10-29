#!/bin/sh
#  dtrsm alone in a process (BENCH3_ONLY=3) vs. after the other routines
cd /tmp/ulmblas-intel2.2yi6R9/bench || exit 1
for v in avx-pre avx-cache avx-cur; do
    gcc -O2 -Iv-$v ../bench3x.c v-$v/libcblas.a -lstdc++ -lm -o bench3x-$v || exit 1
done
for r in 1 2 3 4 5; do
    for v in avx-pre avx-cache avx-cur; do
        BENCH3_ONLY=3 taskset -c 3 ./bench3x-$v 5 512 1024 > res/trsm-only-$v-$r.txt
    done
done
echo end > res/trsm-only.done
