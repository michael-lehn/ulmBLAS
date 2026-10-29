#ifndef CULMBLAS_STEP12_GEMM_H
#define CULMBLAS_STEP12_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 12 of the tutorial: step 11 with three kernels, 2 x 2, 4 x 2 and
//  4 x 4 blocks of 8 x 8 elements of C per warp; gemm chooses one by the
//  size of C (see README.md).  Needs compute capability 8.0 or higher.
//  Same interface and semantics as cULMblas::gemm (culmblas/gemm.h).
//

#ifndef CULMBLAS_STEP12_WARPS_M
#define CULMBLAS_STEP12_WARPS_M 2
#endif
#ifndef CULMBLAS_STEP12_WARPS_N
#define CULMBLAS_STEP12_WARPS_N 2
#endif
#ifndef CULMBLAS_STEP12_STAGES
#define CULMBLAS_STEP12_STAGES 2
#endif
#ifndef CULMBLAS_STEP12_GROUP
#define CULMBLAS_STEP12_GROUP 8
#endif

namespace cULMblas { namespace step12 {

constexpr int TileM      = 8;           // mma.sync for double:
constexpr int TileN      = 8;           // C(8 x 8) += A(8 x 4) B(4 x 8)
constexpr int TileK      = 4;
constexpr int WarpsM     = CULMBLAS_STEP12_WARPS_M;     // per block
constexpr int WarpsN     = CULMBLAS_STEP12_WARPS_N;
constexpr int Stages     = CULMBLAS_STEP12_STAGES;      // panel buffers
constexpr int Group      = CULMBLAS_STEP12_GROUP;       // block rows

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream = 0);

} } // namespace step12, cULMblas

#endif // CULMBLAS_STEP12_GEMM_H
