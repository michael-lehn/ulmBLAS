//
//  Step 07 of the tutorial (experimental): several 8 x 8 blocks of C per
//  warp.
//
//  A warp computes TilesM x TilesN blocks of 8 x 8 elements of C, i.e.
//  WarpRows x WarpCols elements.  For KSteps steps of 4 values of l the 32
//  threads of the warp copy the WarpRows x 4*KSteps block of A and the
//  4*KSteps x WarpCols block of B into its buffer in shared memory (0
//  outside of A and B), then for every step of 4 the warp loads TilesM
//  fragments of A and TilesN fragments of B and does TilesM x TilesN
//  mma_sync: every fragment of A is used TilesN times, every fragment of B
//  TilesM times.
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step07/gemm.h>

#include <mma.h>

#include <limits>
#include <stdexcept>

namespace cULMblas { namespace step07 {

namespace {

constexpr int NumWarps = WarpsM*WarpsN;
constexpr int PanelK   = KSteps*TileK;                  // values of l per copy
constexpr int SizeA    = WarpRows*PanelK;               // per warp, doubles
constexpr int SizeB    = PanelK*WarpCols;
constexpr int SizeC    = WarpRows*WarpCols;
constexpr int SizeBuf  = SizeA + SizeB > SizeC ? SizeA + SizeB : SizeC;

static_assert(SizeA % 32 == 0 && SizeB % 32 == 0,
              "the threads of a warp copy the blocks in equal parts");

std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

#ifndef CULMBLAS_STEP07_MIN_BLOCKS
#define CULMBLAS_STEP07_MIN_BLOCKS 0
#endif
#if CULMBLAS_STEP07_MIN_BLOCKS
__global__ void __launch_bounds__(32*NumWarps, CULMBLAS_STEP07_MIN_BLOCKS)
#else
__global__ void __launch_bounds__(32*NumWarps)
#endif
dgemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
      double alpha,
      const double *A, std::ptrdiff_t ldA,
      const double *B, std::ptrdiff_t ldB,
      double beta,
      double *C, std::ptrdiff_t ldC)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;

    //  per warp: the blocks of A (column major, WarpRows x PanelK) and B
    //  (column major, PanelK x WarpCols), at the end the block of C
    __shared__ __align__(32) double buf[NumWarps][SizeBuf];

    const int lane = threadIdx.x, warp = threadIdx.y;
    double *sA = buf[warp], *sB = buf[warp] + SizeA, *sC = buf[warp];

    const std::ptrdiff_t i0 = std::ptrdiff_t(blockIdx.x)*BlockRows
                            + (warp % WarpsM)*WarpRows;
    const std::ptrdiff_t j0 = std::ptrdiff_t(blockIdx.y)*BlockCols
                            + (warp / WarpsM)*WarpCols;

    if (i0 >= m || j0 >= n) {
        return;
    }

    wmma::fragment<wmma::accumulator, TileM, TileN, TileK, double>
        ab[TilesM][TilesN];
    for (int r = 0; r < TilesM; ++r) {
        for (int c = 0; c < TilesN; ++c) {
            wmma::fill_fragment(ab[r][c], 0.0);
        }
    }

    if (alpha != 0) {
        for (std::ptrdiff_t l0 = 0; l0 < k; l0 += PanelK) {
            for (int q = 0; q < SizeA/32; ++q) {
                int e = lane + 32*q, i = e % WarpRows, l = e / WarpRows;
                sA[e] = (i0 + i < m && l0 + l < k)
                      ? A[i0 + i + (l0 + l)*ldA] : 0;
            }
            for (int q = 0; q < SizeB/32; ++q) {
                int e = lane + 32*q, l = e % PanelK, j = e / PanelK;
                sB[e] = (l0 + l < k && j0 + j < n)
                      ? B[l0 + l + (j0 + j)*ldB] : 0;
            }
            __syncwarp();

            #pragma unroll
            for (int s = 0; s < KSteps; ++s) {
                wmma::fragment<wmma::matrix_a, TileM, TileN, TileK, double,
                               wmma::col_major> a[TilesM];
                wmma::fragment<wmma::matrix_b, TileM, TileN, TileK, double,
                               wmma::col_major> b[TilesN];
                for (int r = 0; r < TilesM; ++r) {
                    wmma::load_matrix_sync(a[r],
                                           sA + r*TileM + s*TileK*WarpRows,
                                           WarpRows);
                }
                for (int c = 0; c < TilesN; ++c) {
                    wmma::load_matrix_sync(b[c],
                                           sB + s*TileK + c*TileN*PanelK,
                                           PanelK);
                }
                for (int r = 0; r < TilesM; ++r) {
                    for (int c = 0; c < TilesN; ++c) {
                        wmma::mma_sync(ab[r][c], a[r], b[c], ab[r][c]);
                    }
                }
            }
            __syncwarp();
        }
    }

    for (int r = 0; r < TilesM; ++r) {
        for (int c = 0; c < TilesN; ++c) {
            wmma::store_matrix_sync(sC + r*TileM + c*TileN*WarpRows,
                                    ab[r][c], WarpRows, wmma::mem_col_major);
        }
    }
    __syncwarp();
    for (int e = lane; e < SizeC; e += 32) {
        std::ptrdiff_t i = i0 + e % WarpRows, j = j0 + e / WarpRows;
        if (i < m && j < n) {
            double &cij = C[i + j*ldC];
            cij = (beta == 0) ? alpha*sC[e] : beta*cij + alpha*sC[e];
        }
    }
#endif
}

bool
haveFP64TensorCores()
{
    int device, major;
    CULMBLAS_CHECK_CUDA(cudaGetDevice(&device));
    CULMBLAS_CHECK_CUDA(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device));
    return major >= 8;
}

} // namespace

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream)
{
    checkGemmArguments(m, n, k, A, ldA, B, ldB, C, ldC);

    if (m == 0 || n == 0 || ((alpha == 0 || k == 0) && beta == 1)) {
        return;
    }
    static const bool tensorCores = haveFP64TensorCores();
    if (!tensorCores) {
        throw std::runtime_error("cULMblas::step07::gemm: needs a GPU with "
                                 "FP64 Tensor Cores (compute capability "
                                 ">= 8.0)");
    }

    std::ptrdiff_t numBlocksM = ceildiv(m, BlockRows);
    std::ptrdiff_t numBlocksN = ceildiv(n, BlockCols);

    if (numBlocksM > std::numeric_limits<int>::max() || numBlocksN > 65535) {
        throw std::invalid_argument("cULMblas::step07::gemm: m or n too "
                                    "large for the grid of this kernel");
    }

    dim3 block(32, NumWarps);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    dgemm<<<grid, block, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB,
                                      beta, C, ldC);
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step07, cULMblas
