#ifndef CULMBLAS_STEP14_GEMM_H
#define CULMBLAS_STEP14_GEMM_H 1

#include <cstddef>
#include <cuda_runtime.h>

//
//  Step 14 of the tutorial: step 13 with the operands of a panel loaded
//  in groups of 8 values of l, and the number of panel buffers and the
//  panel length as parameters of each kernel.  With the defaults (the
//  configuration of step 13) as fast as step 13; the other settings are
//  the experiments of this step, none of them reproducibly faster (see
//  README.md).  Needs compute capability 8.0 or higher.  Same interface
//  and semantics as cULMblas::gemm (culmblas/gemm.h).
//

#ifndef CULMBLAS_STEP14_WARPS_M
#define CULMBLAS_STEP14_WARPS_M 2
#endif
#ifndef CULMBLAS_STEP14_WARPS_N
#define CULMBLAS_STEP14_WARPS_N 2
#endif
#ifndef CULMBLAS_STEP14_GROUP
#define CULMBLAS_STEP14_GROUP 8
#endif
//  values of l per panel and panel buffers of the three kernels
#ifndef CULMBLAS_STEP14_PANEL_SMALL
#define CULMBLAS_STEP14_PANEL_SMALL 16
#endif
#ifndef CULMBLAS_STEP14_PANEL_MEDIUM
#define CULMBLAS_STEP14_PANEL_MEDIUM 16
#endif
#ifndef CULMBLAS_STEP14_PANEL_LARGE
#define CULMBLAS_STEP14_PANEL_LARGE 8
#endif
#ifndef CULMBLAS_STEP14_STAGES_SMALL
#define CULMBLAS_STEP14_STAGES_SMALL 2
#endif
#ifndef CULMBLAS_STEP14_STAGES_MEDIUM
#define CULMBLAS_STEP14_STAGES_MEDIUM 2
#endif
#ifndef CULMBLAS_STEP14_STAGES_LARGE
#define CULMBLAS_STEP14_STAGES_LARGE 2
#endif
//  blocks per SM the registers must allow (0: no such requirement)
#ifndef CULMBLAS_STEP14_MIN_BLOCKS
#define CULMBLAS_STEP14_MIN_BLOCKS 0
#endif

namespace cULMblas { namespace step14 {

constexpr int TileM      = 8;           // mma.sync for double:
constexpr int TileN      = 8;           // C(8 x 8) += A(8 x 4) B(4 x 8)
constexpr int TileK      = 4;
constexpr int WarpsM     = CULMBLAS_STEP14_WARPS_M;     // per block
constexpr int WarpsN     = CULMBLAS_STEP14_WARPS_N;
constexpr int Group      = CULMBLAS_STEP14_GROUP;       // block rows
constexpr int PanelSmall   = CULMBLAS_STEP14_PANEL_SMALL;
constexpr int PanelMedium  = CULMBLAS_STEP14_PANEL_MEDIUM;
constexpr int PanelLarge   = CULMBLAS_STEP14_PANEL_LARGE;
constexpr int StagesSmall  = CULMBLAS_STEP14_STAGES_SMALL;
constexpr int StagesMedium = CULMBLAS_STEP14_STAGES_MEDIUM;
constexpr int StagesLarge  = CULMBLAS_STEP14_STAGES_LARGE;

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream = 0);

} } // namespace step14, cULMblas

#endif // CULMBLAS_STEP14_GEMM_H
