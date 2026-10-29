#ifndef CULMBLAS_GEMM_H
#define CULMBLAS_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  cULMblas: GEMM with CUDA, one tutorial step at a time.
//
//      C <- beta*C + alpha*A*B
//
//  - double precision, A is m x k, B is k x n, C is m x n;
//  - column major, as in BLAS and cuBLAS: element (i, j) of A is
//    A[i + j*ldA], with ldA >= max(1, m), ldB >= max(1, k),
//    ldC >= max(1, m); no transposition;
//  - A, B and C are device pointers (memory allocated with cudaMalloc);
//    the host only launches the kernel;
//  - the function is asynchronous with respect to the host, like cuBLAS:
//    it enqueues the work on `stream` and returns;
//  - BLAS semantics of alpha and beta: if alpha == 0 or k == 0, A and B are
//    not read; if beta == 0, C is not read (it may contain NaN).
//
//  Invalid arguments throw std::invalid_argument, a failed launch throws
//  std::runtime_error.  Errors of the kernel itself are reported by the
//  next synchronizing CUDA call.
//
//  cULMblas::gemm is the current implementation.  Every tutorial step stays
//  available in its own directory and namespace (step01/, cULMblas::step01,
//  ...), so that the tests and the benchmark can compare all of them.
//

namespace cULMblas {

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream = 0);

//
//  Checks of the arguments shared by all implementations (throws
//  std::invalid_argument).
//
void
checkGemmArguments(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
                   const double *A, std::ptrdiff_t ldA,
                   const double *B, std::ptrdiff_t ldB,
                   const double *C, std::ptrdiff_t ldC);

} // namespace cULMblas

#endif // CULMBLAS_GEMM_H
