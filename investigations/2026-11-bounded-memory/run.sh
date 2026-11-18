#!/bin/sh
#
#  Benchmarks of potrf, getrf and getri (bench.cc) for a source tree:
#
#      run.sh srcdir outfile [openblas.so]
#
#  Builds lapackbench (serial, without OpenMP) and lapackbench-omp (OpenMP
#  tasks)
#  with the libraries of srcdir (make cblas, OPENMP=0 and 1: srcdir and
#  srcdir-omp) and runs 1 and 6 threads, unit stride (s = 1) and general
#  strides (s = 2, 42).  OpenBLAS only for s = 1.  ROUTINES: the routines
#  (default: potrf getrf getri).
#
set -e
src=$1
out=$2
ob=$3
here=$(cd "$(dirname "$0")" && pwd)

g++ -O3 -march=native -std=c++11 -I"$src" -o "$src/lapackbench" "$here/bench.cc" \
    "$src/libcblas.a" -pthread -ldl
g++ -O3 -march=native -std=c++11 -fopenmp -I"$src-omp" -o "$src/lapackbench-omp" \
    "$here/bench.cc" "$src-omp/libcblas.a" -pthread -ldl

: > "$out"
for t in 1 6; do
    export ULMBLAS_NUM_THREADS=$t OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t
    export OMP_MAX_TASK_PRIORITY=3 OMP_PROC_BIND=close OMP_PLACES=cores
    if [ $t = 1 ]; then b="$src/lapackbench"; else b="$src/lapackbench-omp"; fi
    for r in ${ROUTINES:-potrf getrf getri}; do
	for n in 500 1000 2000 4000; do
	    for s in 1 2 42; do
		if [ $s = 42 ] && [ $n -gt 2000 ]; then continue; fi
		reps=3
		if [ $n -ge 4000 ]; then reps=2; fi
		echo "threads=$t $($b $r $n $s $reps)" >> "$out"
		if [ $s = 1 ] && [ -n "$ob" ]; then
		    echo "threads=$t $($b $r $n $s $reps $ob)" >> "$out"
		fi
	    done
	done
    done
done
