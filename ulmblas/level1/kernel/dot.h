#ifndef ULMBLAS_LEVEL1_KERNEL_DOT_H
#define ULMBLAS_LEVEL1_KERNEL_DOT_H 1

//
//  dot: the reference kernel on all platforms, vectorized with partial sums
//  (ulmblas/level1/kernel/reduce.h).  The SSE kernel of before
//  (sse/dot.h, double only, two elements per instruction) is slower: on an
//  Ice Lake core ddot n = 1000: 1.9 elements per ns against 8.6 (OpenBLAS
//  6.9); n = 100000: 1.7 against 3.2 (OpenBLAS 2.7).
//
#define SELECT_DOT_KERNEL ref
#include <ulmblas/level1/kernel/ref/dot.h>

#endif // ULMBLAS_LEVEL1_KERNEL_DOT_H
