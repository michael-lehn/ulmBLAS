#ifndef CULMBLAS_STEP01_GEMM_H
#define CULMBLAS_STEP01_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 01 of the tutorial: the naive GEMM, one thread per element of C,
//  32 x 32 threads per block (see README.md).  Same interface and
//  semantics as cULMblas::gemm (culmblas/gemm.h).
//

namespace cULMblas { namespace step01 {

constexpr int BlockDim = 32;

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream = 0);

} } // namespace step01, cULMblas

#endif // CULMBLAS_STEP01_GEMM_H
