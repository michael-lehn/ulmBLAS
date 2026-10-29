#ifndef CULMBLAS_STEP04_GEMM_H
#define CULMBLAS_STEP04_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 04 of the tutorial: as step 03, a thread computes a ThreadRows x
//  ThreadCols block of C with rows 32 apart, but the loop over l is done in
//  passes of Pass values of l: first all loads of a pass, then all its
//  multiply-adds (see README.md).  Same interface and semantics as
//  cULMblas::gemm (culmblas/gemm.h).
//

#ifndef CULMBLAS_STEP04_THREAD_ROWS
#define CULMBLAS_STEP04_THREAD_ROWS 4
#endif

#ifndef CULMBLAS_STEP04_THREAD_COLS
#define CULMBLAS_STEP04_THREAD_COLS 4
#endif

#ifndef CULMBLAS_STEP04_PASS
#define CULMBLAS_STEP04_PASS 4
#endif

namespace cULMblas { namespace step04 {

constexpr int ThreadsX   = 32;                              // threadIdx.x
constexpr int ThreadsY   = 8;                               // threadIdx.y
constexpr int ThreadRows = CULMBLAS_STEP04_THREAD_ROWS;     // per thread
constexpr int ThreadCols = CULMBLAS_STEP04_THREAD_COLS;     // per thread
constexpr int Pass       = CULMBLAS_STEP04_PASS;            // values of l
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

} } // namespace step04, cULMblas

#endif // CULMBLAS_STEP04_GEMM_H
