//
//  ulmBLAS: declaration of the DGEMM kernel of OpenBLAS 0.3.34 for HASWELL
//  (AVX2/FMA, kernel/x86_64/dgemm_kernel_4x8_haswell.S), assembled from
//  ../asm/haswell_4x8.S as the function
//  ulmblas_openblas_dgemm_kernel_haswell_4x8 (see ../README.md).  The
//  libraries of ulmBLAS contain the object file if this kernel is selected
//  (make.inc, ULMBLAS_ASM_SOURCES); programs that use the C++ templates
//  without these libraries must assemble ../asm/haswell_4x8.S themselves.
//
//  The function is OpenBLAS's GEMM kernel interface: for a packed block
//  (A in panels of 4 rows, B in panels of 8 columns, both k-major, as
//  ulmBLAS packs them for MR = 4, NR = 8) it computes C <- C + alpha*A*B
//  for column-major C with leading dimension ldc.  It copies 12 columns of
//  B at a time into a buffer of 12*k doubles on its stack (L_BUFFER_SIZE:
//  256*8*12 + 4096 bytes), so k must not exceed 298 (KMAX in select.h).
//
#ifndef ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_KERNEL_HASWELL_4X8_H
#define ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_KERNEL_HASWELL_4X8_H 1

#include <ulmblas/level3/ukernel/openblas/compat.h>

extern "C" int
ulmblas_openblas_dgemm_kernel_haswell_4x8(long m, long n, long k, double alpha,
                                          double *A, double *B, double *C,
                                          long ldc);

namespace ulmBLAS { namespace openblas { namespace kernel {
namespace haswell_4x8 {

using namespace ulmBLAS::openblas::compat;

inline int
dgemm_kernel(BLASLONG m, BLASLONG n, BLASLONG k, double alpha,
             double *A, double *B, double *C, BLASLONG ldc)
{
    return ulmblas_openblas_dgemm_kernel_haswell_4x8(m, n, k, alpha,
                                                     A, B, C, ldc);
}

} } } } // namespace haswell_4x8, kernel, openblas, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_KERNEL_HASWELL_4X8_H
