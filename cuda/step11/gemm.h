#ifndef CULMBLAS_STEP11_GEMM_H
#define CULMBLAS_STEP11_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 11 of the tutorial: step 10 with another order of the blocks: the
//  blocks running at the same time cover groups of Group block rows of C
//  and share their panels of A and B in the L2 cache (see README.md).
//  Needs compute capability 8.0 or higher.  Same interface and semantics
//  as cULMblas::gemm (culmblas/gemm.h).
//

#ifndef CULMBLAS_STEP11_TILES_M
#define CULMBLAS_STEP11_TILES_M 4
#endif
#ifndef CULMBLAS_STEP11_TILES_N
#define CULMBLAS_STEP11_TILES_N 2
#endif
#ifndef CULMBLAS_STEP11_WARPS_M
#define CULMBLAS_STEP11_WARPS_M 2
#endif
#ifndef CULMBLAS_STEP11_WARPS_N
#define CULMBLAS_STEP11_WARPS_N 2
#endif
#ifndef CULMBLAS_STEP11_STAGES
#define CULMBLAS_STEP11_STAGES 2
#endif
#ifndef CULMBLAS_STEP11_GROUP
#define CULMBLAS_STEP11_GROUP 8
#endif
#ifndef CULMBLAS_STEP11_PANEL
#define CULMBLAS_STEP11_PANEL 16
#endif

namespace cULMblas { namespace step11 {

constexpr int TileM      = 8;           // mma.sync for double:
constexpr int TileN      = 8;           // C(8 x 8) += A(8 x 4) B(4 x 8)
constexpr int TileK      = 4;
constexpr int TilesM     = CULMBLAS_STEP11_TILES_M;     // per warp
constexpr int TilesN     = CULMBLAS_STEP11_TILES_N;
constexpr int WarpsM     = CULMBLAS_STEP11_WARPS_M;     // per block
constexpr int WarpsN     = CULMBLAS_STEP11_WARPS_N;
constexpr int Panel      = CULMBLAS_STEP11_PANEL;       // values of l
constexpr int Stages     = CULMBLAS_STEP11_STAGES;      // panel buffers
constexpr int Group      = CULMBLAS_STEP11_GROUP;       // block rows
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

} } // namespace step11, cULMblas

#endif // CULMBLAS_STEP11_GEMM_H
