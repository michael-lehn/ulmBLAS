#!/bin/sh
#  run.sh BENCH LIB NAME: all operations and sizes on 1 and 6 threads,
#  lines "NAME THREADS OP N GFLOPS"
bench=$1
lib=$2
name=$3
ops=${OPS:-"dgemv_n dgemv_t sgemv_n sgemv_t zgemv_n zgemv_t zgemv_c dger zgeru
dtrsv_ln dtrsv_lt dtrsv_un dtrsv_ut ztrsv_ln ztrsv_lt ztrsv_un ztrsv_ut
dgemv_rn dgemv_rt"}
sizes=${SIZES:-"100 500 1000 2000 4000 8000"}
for t in ${THREADS:-1 6}; do
    for op in $ops; do
	OPENBLAS_NUM_THREADS=$t ULMBLAS_NUM_THREADS=$t OMP_NUM_THREADS=$t \
	    $bench $lib $op $sizes | sed "s/^/$name $t /"
    done
done
