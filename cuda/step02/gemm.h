#ifndef CULMBLAS_STEP02_GEMM_H
#define CULMBLAS_STEP02_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 02 of the tutorial: each thread computes ThreadCols consecutive
//  elements of a row of C and loads each element of A only once for them
//  (see README.md).  Same interface and semantics as cULMblas::gemm
//  (culmblas/gemm.h).
//

#ifndef CULMBLAS_STEP02_THREAD_COLS
#define CULMBLAS_STEP02_THREAD_COLS 4
#endif

namespace cULMblas { namespace step02 {

constexpr int BlockRows  = 32;                              // threadIdx.x
constexpr int ThreadCols = CULMBLAS_STEP02_THREAD_COLS;     // per thread
constexpr int BlockCols  = 32;                              // of C
constexpr int ThreadsY   = BlockCols/ThreadCols;            // threadIdx.y

static_assert(BlockCols % ThreadCols == 0,
              "ThreadCols must divide BlockCols");

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream = 0);

} } // namespace step02, cULMblas

#endif // CULMBLAS_STEP02_GEMM_H
