#ifndef CULMBLAS_STEP03_GEMM_H
#define CULMBLAS_STEP03_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 03 of the tutorial: each thread computes a ThreadRows x ThreadCols
//  block of C, with rows 32 apart, and loads each element of A and B only
//  once for it (see README.md).  Same interface and semantics as
//  cULMblas::gemm (culmblas/gemm.h).
//

#ifndef CULMBLAS_STEP03_THREAD_ROWS
#define CULMBLAS_STEP03_THREAD_ROWS 8
#endif

#ifndef CULMBLAS_STEP03_THREAD_COLS
#define CULMBLAS_STEP03_THREAD_COLS 4
#endif

namespace cULMblas { namespace step03 {

constexpr int ThreadsX   = 32;                              // threadIdx.x
constexpr int ThreadsY   = 8;                               // threadIdx.y
constexpr int ThreadRows = CULMBLAS_STEP03_THREAD_ROWS;     // per thread
constexpr int ThreadCols = CULMBLAS_STEP03_THREAD_COLS;     // per thread
constexpr int BlockRows  = ThreadsX*ThreadRows;             // of C
constexpr int BlockCols  = ThreadsY*ThreadCols;             // of C

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream = 0);

} } // namespace step03, cULMblas

#endif // CULMBLAS_STEP03_GEMM_H
