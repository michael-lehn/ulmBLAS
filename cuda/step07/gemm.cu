//
//  Step 07 of the tutorial: several 8 x 8 blocks of C per warp.
//
//  As in step 06 the 32 threads of a warp copy the blocks of A and B they
//  need for a step of 4 values of l into their buffer in shared memory (0
//  outside of A and B) and load the fragments from there.  But a warp now
//  computes TilesM x TilesN blocks of 8 x 8 elements of C (default 4 x 4,
//  32 x 32 elements): per step it copies a 32 x 4 block of A and a 4 x 32
//  block of B, loads 4 fragments of A and 4 of B and does 16 mma_sync.
//  Every fragment of A is used for 4 blocks of C, every fragment of B as
//  well; in step 06 every fragment was used once.
//
//  A block of WarpsM x WarpsN warps (default 2 x 2, 128 threads) covers
//  64 x 64 elements of C.
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
constexpr int SizeA    = WarpRows*TileK;                // per warp, doubles
constexpr int SizeB    = TileK*WarpCols;
constexpr int SizeC    = WarpRows*WarpCols;
constexpr int SizeBuf  = SizeA + SizeB > SizeC ? SizeA + SizeB : SizeC;

static_assert(SizeA % 32 == 0 && SizeB % 32 == 0,
              "the threads of a warp copy the blocks in equal parts");

//  ceil(a/b) for a >= 0, b > 0, without the overflow of (a + b - 1)/b
std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

//  threadIdx.x: the thread in its warp (0..31), threadIdx.y: the warp
__global__ void __launch_bounds__(32*NumWarps)
dgemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
      double alpha,
      const double *A, std::ptrdiff_t ldA,
      const double *B, std::ptrdiff_t ldB,
      double beta,
      double *C, std::ptrdiff_t ldC)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;

    //  per warp, column major: the WarpRows x 4 block of A and the
    //  4 x WarpCols block of B; at the end the WarpRows x WarpCols block of C
    __shared__ __align__(32) double buf[NumWarps][SizeBuf];

    const int lane = threadIdx.x, warp = threadIdx.y;
    double *sA = buf[warp], *sB = buf[warp] + SizeA, *sC = buf[warp];

    const std::ptrdiff_t i0 = std::ptrdiff_t(blockIdx.x)*BlockRows
                            + (warp % WarpsM)*WarpRows;
    const std::ptrdiff_t j0 = std::ptrdiff_t(blockIdx.y)*BlockCols
                            + (warp / WarpsM)*WarpCols;

    //  the whole warp leaves (the WMMA operations need all its threads)
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
        for (std::ptrdiff_t l = 0; l < k; l += TileK) {
            //  element e of the block of A: row e % WarpRows, column
            //  e / WarpRows; of the block of B: row e % 4, column e / 4
            for (int q = 0; q < SizeA/32; ++q) {
                int e = lane + 32*q, i = e % WarpRows, p = e / WarpRows;
                sA[e] = (i0 + i < m && l + p < k)
                      ? A[i0 + i + (l + p)*ldA] : 0;
            }
            for (int q = 0; q < SizeB/32; ++q) {
                int e = lane + 32*q, p = e % TileK, j = e / TileK;
                sB[e] = (l + p < k && j0 + j < n)
                      ? B[l + p + (j0 + j)*ldB] : 0;
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, TileM, TileN, TileK, double,
                           wmma::col_major> a[TilesM];
            wmma::fragment<wmma::matrix_b, TileM, TileN, TileK, double,
                           wmma::col_major> b[TilesN];
            for (int r = 0; r < TilesM; ++r) {
                wmma::load_matrix_sync(a[r], sA + r*TileM, WarpRows);
            }
            for (int c = 0; c < TilesN; ++c) {
                wmma::load_matrix_sync(b[c], sB + c*TileN*TileK, TileK);
            }
            for (int r = 0; r < TilesM; ++r) {
                for (int c = 0; c < TilesN; ++c) {
                    wmma::mma_sync(ab[r][c], a[r], b[c], ab[r][c]);
                }
            }
            __syncwarp();
        }
    }

    //  C = beta*C + alpha*AB for the elements inside of C (beta == 0: C is
    //  not read, it may contain NaN)
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

//  compute capability of the current device >= 8.0?
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
