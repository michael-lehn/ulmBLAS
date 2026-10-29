#ifndef CULMBLAS_XMAGMA_GEMM_H
#define CULMBLAS_XMAGMA_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Experiment (not part of the tutorial): the algorithm of magmablas_dgemm
//  (NN, CUDA) rebuilt step by step, and simplified towards step 03/04.
//
//      v0  64 x 64 tile, 16 x 16 threads, 4 x 4 per thread (stride 16),
//          panels of 16 in shared memory, next panel prefetched in registers
//      v1  as v0 without the prefetch
//      v2  no shared memory: the 4 x 4 per thread straight from global
//          memory, loop over l in passes of 16 (no k checks inside a pass)
//      v3  as v2 with passes of 4
//      v4  as v3 with 32 x 8 threads: rows 32 apart, 4 neighboring columns
//          (step 04 4x4, passes of 4)
//
namespace cULMblas { namespace xmagma {

#define CULMBLAS_XMAGMA_DECL(v) \
void gemm##v(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,        \
             double alpha, const double *A, std::ptrdiff_t ldA,            \
             const double *B, std::ptrdiff_t ldB, double beta,             \
             double *C, std::ptrdiff_t ldC, cudaStream_t stream = 0);
CULMBLAS_XMAGMA_DECL(0)
CULMBLAS_XMAGMA_DECL(1)
CULMBLAS_XMAGMA_DECL(2)
CULMBLAS_XMAGMA_DECL(3)
CULMBLAS_XMAGMA_DECL(4)
CULMBLAS_XMAGMA_DECL(5)
CULMBLAS_XMAGMA_DECL(6)
CULMBLAS_XMAGMA_DECL(7)
CULMBLAS_XMAGMA_DECL(8)
CULMBLAS_XMAGMA_DECL(9)
CULMBLAS_XMAGMA_DECL(10)
CULMBLAS_XMAGMA_DECL(11)
CULMBLAS_XMAGMA_DECL(12)
CULMBLAS_XMAGMA_DECL(13)
#undef CULMBLAS_XMAGMA_DECL

} } // namespace xmagma, cULMblas

#endif
