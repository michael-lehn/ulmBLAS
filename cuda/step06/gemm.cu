//
//  Step 06 of the tutorial: the first GEMM with the Tensor Cores.
//
//  A Tensor Core instruction is executed by a whole warp: for double
//  precision it computes C(8 x 8) += A(8 x 4) B(4 x 8), where the 32 threads
//  of the warp hold the operands together in their registers.  CUDA's WMMA
//  interface (namespace nvcuda::wmma) hides how the elements are spread
//  over the threads: load_matrix_sync loads a "fragment" (A, B or C) from
//  memory, mma_sync does the multiply-add, store_matrix_sync stores the
//  result.
//
//  Here every warp computes one 8 x 8 block of C, the loop over l goes in
//  steps of 4.  For every step the 32 threads of the warp copy the 8 x 4
//  block of A and the 4 x 8 block of B into shared memory, each thread one
//  element of A and one of B, with 0 for elements outside of A and B.  The
//  fragments are loaded from there.  That handles the borders of the
//  matrices and the rules of load_matrix_sync for the address and the
//  leading dimension (32-byte aligned, a multiple of 16 bytes) for all
//  matrices; it is the simple, not the fast way.
//
//  A block of WarpsM x WarpsN warps (default 2 x 2, 128 threads) covers
//  16 x 16 elements of C.
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step06/gemm.h>

#include <mma.h>

#include <limits>
#include <stdexcept>

namespace cULMblas { namespace step06 {

namespace {

constexpr int NumWarps = WarpsM*WarpsN;

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

    //  per warp, column major: an 8 x 4 block of A, a 4 x 8 block of B and
    //  the 8 x 8 block of C
    __shared__ __align__(32) double sA[NumWarps][TileK*TileM];
    __shared__ __align__(32) double sB[NumWarps][TileN*TileK];
    __shared__ __align__(32) double sC[NumWarps][TileN*TileM];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const std::ptrdiff_t i0 = std::ptrdiff_t(blockIdx.x)*BlockRows
                            + (warp % WarpsM)*TileM;
    const std::ptrdiff_t j0 = std::ptrdiff_t(blockIdx.y)*BlockCols
                            + (warp / WarpsM)*TileN;

    //  the whole warp leaves (the WMMA operations need all its threads)
    if (i0 >= m || j0 >= n) {
        return;
    }

    //  the element of A (row iA, column lA) and of B (row lB, column jB)
    //  this thread copies
    const int iA = lane % TileM, lA = lane / TileM;
    const int lB = lane % TileK, jB = lane / TileK;

    wmma::fragment<wmma::accumulator, TileM, TileN, TileK, double> ab;
    wmma::fill_fragment(ab, 0.0);

    if (alpha != 0) {
        for (std::ptrdiff_t l = 0; l < k; l += TileK) {
            sA[warp][iA + lA*TileM]
                = (i0 + iA < m && l + lA < k) ? A[i0 + iA + (l + lA)*ldA] : 0;
            sB[warp][lB + jB*TileK]
                = (l + lB < k && j0 + jB < n) ? B[l + lB + (j0 + jB)*ldB] : 0;
            __syncwarp();

            wmma::fragment<wmma::matrix_a, TileM, TileN, TileK, double,
                           wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, TileM, TileN, TileK, double,
                           wmma::col_major> b;
            wmma::load_matrix_sync(a, sA[warp], TileM);
            wmma::load_matrix_sync(b, sB[warp], TileK);
            wmma::mma_sync(ab, a, b, ab);
            __syncwarp();
        }
    }

    //  C = beta*C + alpha*AB for the elements inside of C (beta == 0: C is
    //  not read, it may contain NaN)
    wmma::store_matrix_sync(sC[warp], ab, TileM, wmma::mem_col_major);
    __syncwarp();
    for (int e = lane; e < TileM*TileN; e += 32) {
        std::ptrdiff_t i = i0 + e % TileM, j = j0 + e / TileM;
        if (i < m && j < n) {
            double &cij = C[i + j*ldC];
            cij = (beta == 0) ? alpha*sC[warp][e]
                              : beta*cij + alpha*sC[warp][e];
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
        throw std::runtime_error("cULMblas::step06::gemm: needs a GPU with "
                                 "FP64 Tensor Cores (compute capability "
                                 ">= 8.0)");
    }

    std::ptrdiff_t numBlocksM = ceildiv(m, BlockRows);
    std::ptrdiff_t numBlocksN = ceildiv(n, BlockCols);

    if (numBlocksM > std::numeric_limits<int>::max() || numBlocksN > 65535) {
        throw std::invalid_argument("cULMblas::step06::gemm: m or n too "
                                    "large for the grid of this kernel");
    }

    dim3 block(32, NumWarps);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    dgemm<<<grid, block, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB,
                                      beta, C, ldC);
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step06, cULMblas
