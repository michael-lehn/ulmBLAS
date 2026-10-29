#ifndef CULMBLAS_STEP05_GEMM_H
#define CULMBLAS_STEP05_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 05 of the tutorial (experimental): panels of A and B in shared
//  memory.  Same interface and semantics as cULMblas::gemm
//  (culmblas/gemm.h).
//

#ifndef CULMBLAS_STEP05_THREADS_X
#define CULMBLAS_STEP05_THREADS_X 32
#endif
#ifndef CULMBLAS_STEP05_THREADS_Y
#define CULMBLAS_STEP05_THREADS_Y 8
#endif
#ifndef CULMBLAS_STEP05_THREAD_ROWS
#define CULMBLAS_STEP05_THREAD_ROWS 4
#endif
#ifndef CULMBLAS_STEP05_THREAD_COLS
#define CULMBLAS_STEP05_THREAD_COLS 4
#endif
#ifndef CULMBLAS_STEP05_PANEL
#define CULMBLAS_STEP05_PANEL 16
#endif

namespace cULMblas { namespace step05 {

constexpr int ThreadsX   = CULMBLAS_STEP05_THREADS_X;       // threadIdx.x
constexpr int ThreadsY   = CULMBLAS_STEP05_THREADS_Y;       // threadIdx.y
constexpr int ThreadRows = CULMBLAS_STEP05_THREAD_ROWS;     // per thread
constexpr int ThreadCols = CULMBLAS_STEP05_THREAD_COLS;     // per thread
constexpr int Panel      = CULMBLAS_STEP05_PANEL;           // values of l
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

} } // namespace step05, cULMblas

#endif // CULMBLAS_STEP05_GEMM_H
