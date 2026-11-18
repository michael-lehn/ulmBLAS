#ifndef ULMBLAS_LEVEL1_KERNEL_AXPY_H
#define ULMBLAS_LEVEL1_KERNEL_AXPY_H 1

//
//  axpy: the reference kernel on all platforms (the compiler vectorizes the
//  loop for contiguous vectors).  The SSE kernel of before (double only,
//  two elements per instruction) is slower: see ref/axpy.tcc.
//
#define SELECT_AXPY_KERNEL ref
#include <ulmblas/level1/kernel/ref/axpy.h>

#endif // ULMBLAS_LEVEL1_KERNEL_AXPY_H
