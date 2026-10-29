#ifndef CULMBLAS_STEP06_GEMM_H
#define CULMBLAS_STEP06_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 06 of the tutorial: the first GEMM with the Tensor Cores.  Every
//  warp computes an 8 x 8 block of C with the WMMA operation for double
//  precision, C(8 x 8) += A(8 x 4) B(4 x 8), see README.md.  Needs a GPU
//  with FP64 Tensor Cores (compute capability 8.0 or higher).  Same
//  interface and semantics as cULMblas::gemm (culmblas/gemm.h).
//

#ifndef CULMBLAS_STEP06_WARPS_M
#define CULMBLAS_STEP06_WARPS_M 2
#endif

#ifndef CULMBLAS_STEP06_WARPS_N
#define CULMBLAS_STEP06_WARPS_N 2
#endif

namespace cULMblas { namespace step06 {

constexpr int TileM      = 8;           // a WMMA operation for double:
constexpr int TileN      = 8;           // C(8 x 8) += A(8 x 4) B(4 x 8)
constexpr int TileK      = 4;
constexpr int WarpsM     = CULMBLAS_STEP06_WARPS_M;     // warps per block
constexpr int WarpsN     = CULMBLAS_STEP06_WARPS_N;
constexpr int BlockRows  = WarpsM*TileM;                // of C
constexpr int BlockCols  = WarpsN*TileN;                // of C

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream = 0);

} } // namespace step06, cULMblas

#endif // CULMBLAS_STEP06_GEMM_H
