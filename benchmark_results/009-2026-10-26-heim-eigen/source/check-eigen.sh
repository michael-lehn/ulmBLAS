#!/bin/sh
#  check-eigen.sh WORKSPACE : builds the Eigen programs and records how
#  Eigen is configured and which code it runs (run before bench-eigen.sh).
#
#    - eigen-info: version, SIMD macros, threads, thresholds, blocking
#    - ldd, nm: no BLAS library linked, no BLAS symbol in the binary
#    - objdump: functions with FMA instructions on ymm registers
#    - perf record (task-clock, user space): where the time of the timed
#      loop goes for some sizes, i.e. which product path Eigen takes
#
set -u
W=$1; cd $W || exit 1
E=$W/eigen-5.0.1; L=$W/logs; mkdir -p $L bin
CXXFLAGS="-std=c++17 -O3 -DNDEBUG -march=native -I$E"
: > $L/eigen-build.txt
for p in bench-eigen verify-eigen eigen-info; do
    echo "g++ $CXXFLAGS src/$p.cpp -o bin/$p" >> $L/eigen-build.txt
    g++ $CXXFLAGS src/$p.cpp -o bin/$p >> $L/eigen-build.txt 2>&1 || exit 1
done
{
    echo "g++ -march=native resolves to:"
    g++ -march=native -Q --help=target | grep -E '^ +-march=|^ +-mavx2 |^ +-mfma |^ +-mavx512f '
    echo; ./bin/eigen-info
} > $L/eigen-info.txt 2>&1

{
    echo "== ldd bin/bench-eigen"; ldd bin/bench-eigen
    echo "== BLAS symbols in bin/bench-eigen (nm -C | grep -i -E 'dgemm|cblas|blas_'):"
    nm -C bin/bench-eigen | grep -i -E 'dgemm|cblas|blas_' | grep -v blas_data_mapper || echo "(none)"
    echo "== FMA instructions on ymm registers per function (objdump -d):"
    objdump -d --no-show-raw-insn -C bin/bench-eigen |
        awk '/^[0-9a-f]+ <.*>:/ {f=$0} /vfmadd...pd.*ymm/ {c[f]++}
             END {for (k in c) print c[k], substr(k, 18, 150)}' | sort -rn
} > $L/eigen-binary.txt 2>&1

: > $L/eigen-perf.txt
for n in 4 6 7 8 16 64 512; do
    echo "== n = $n (C.noalias() += A*B): share of task-clock samples" >> $L/eigen-perf.txt
    taskset -c 3 perf record -q -e task-clock -o perf-$n.data ./bin/bench-eigen 3 1 $n $n $n >> $L/eigen-perf.txt 2>&1
    perf report -q -i perf-$n.data --sort symbol --percent-limit 2 2>/dev/null |
        sed 's/  */ /g' | cut -c1-170 >> $L/eigen-perf.txt
done
