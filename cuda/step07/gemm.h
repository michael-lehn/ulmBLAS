#ifndef CULMBLAS_STEP07_GEMM_H
#define CULMBLAS_STEP07_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 07 of the tutorial: a warp computes TilesM x TilesN blocks of
//  8 x 8 elements of C with the Tensor Cores (see README.md).
//  Needs compute capability 8.0 or higher.  Same interface and semantics
//  as cULMblas::gemm (culmblas/gemm.h).
//

#ifndef CULMBLAS_STEP07_TILES_M
#define CULMBLAS_STEP07_TILES_M 4
#endif
#ifndef CULMBLAS_STEP07_TILES_N
#define CULMBLAS_STEP07_TILES_N 4
#endif
#ifndef CULMBLAS_STEP07_WARPS_M
#define CULMBLAS_STEP07_WARPS_M 2
#endif
#ifndef CULMBLAS_STEP07_WARPS_N
#define CULMBLAS_STEP07_WARPS_N 2
#endif

namespace cULMblas { namespace step07 {

constexpr int TileM      = 8;           // a WMMA operation for double:
constexpr int TileN      = 8;           // C(8 x 8) += A(8 x 4) B(4 x 8)
constexpr int TileK      = 4;
constexpr int TilesM     = CULMBLAS_STEP07_TILES_M;     // per warp
constexpr int TilesN     = CULMBLAS_STEP07_TILES_N;
constexpr int WarpsM     = CULMBLAS_STEP07_WARPS_M;     // per block
constexpr int WarpsN     = CULMBLAS_STEP07_WARPS_N;
constexpr int WarpRows   = TilesM*TileM;                // of C, per warp
constexpr int WarpCols   = TilesN*TileN;
constexpr int BlockRows  = WarpsM*WarpRows;             // of C, per block
constexpr int BlockCols  = WarpsN*WarpCols;

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream = 0);

} } // namespace step07, cULMblas

#endif // CULMBLAS_STEP07_GEMM_H
