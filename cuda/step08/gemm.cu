//
//  Step 08 of the tutorial: Tensor Cores with panels of A and B in shared
//  memory for the whole block.
//
//  Step 07 with the panels of step 05: the 128 threads of a block (2 x 2
//  warps) copy a 64 x 16 panel of A and a 16 x 64 panel of B (Panel = 16
//  values of l) into shared memory, each element once, consecutive threads
//  reading consecutive elements of a column, and wait for each other
//  (__syncthreads).  Then every warp loads its fragments from the panels,
//  for each of the 4 steps of 4 values of l: TilesM fragments of A and
//  TilesN of B for its TilesM x TilesN blocks of 8 x 8 elements of C
//  (default 4 x 2), and does TilesM x TilesN mma_sync.
//
//  A block of WarpsM x WarpsN warps covers BlockRows x BlockCols elements
//  of C (default 64 x 32).
//
//  The columns of the panels in shared memory are 4 doubles longer than
//  needed (Pad).  Without that, the fragments loaded by a warp hit the same
//  banks of shared memory, and the loads are serialized (bank conflicts);
//  see README.md.
//
//  Elements outside of A and B are stored as 0 in shared memory; elements
//  of C outside of C are not stored.  Offsets within a panel are int.
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step08/gemm.h>

#include <mma.h>

#include <limits>
#include <stdexcept>

namespace cULMblas { namespace step08 {

namespace {

constexpr int NumWarps   = WarpsM*WarpsN;
constexpr int NumThreads = 32*NumWarps;
constexpr int Pad        = 4;                           // doubles, see above
constexpr int LdA        = BlockRows + Pad;             // of sA
constexpr int LdB        = Panel + Pad;                 // of sB
constexpr int SizeA      = LdA*Panel;
constexpr int SizeB      = LdB*BlockCols;
constexpr int SizeC      = NumWarps*WarpRows*WarpCols;
constexpr int SizeBuf    = SizeA + SizeB > SizeC ? SizeA + SizeB : SizeC;

//  load_matrix_sync: leading dimension a multiple of 16 bytes, addresses
//  32-byte aligned
static_assert(LdA % 2 == 0 && LdB % 2 == 0 && SizeA % 4 == 0,
              "alignment of the fragments in shared memory");
static_assert(BlockRows*Panel % NumThreads == 0,
              "the threads must copy the panel of A in equal parts");
static_assert(Panel*BlockCols % NumThreads == 0,
              "the threads must copy the panel of B in equal parts");
static_assert(Panel % TileK == 0, "a panel is a multiple of 4 values of l");

//  ceil(a/b) for a >= 0, b > 0, without the overflow of (a + b - 1)/b
std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

//  threadIdx.x: the thread in its warp (0..31), threadIdx.y: the warp
__global__ void __launch_bounds__(NumThreads)
dgemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
      double alpha,
      const double *A, int ldA,
      const double *B, int ldB,
      double beta,
      double *C, std::ptrdiff_t ldC)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;

    //  the panels: sA(i,l) = sA[i + l*LdA], sB(l,j) = sB[l + j*LdB]; at the
    //  end the blocks of C of the warps
    __shared__ __align__(32) double buf[SizeBuf];
    double *sA = buf, *sB = buf + SizeA;

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid = warp*32 + lane;
    const int wi = (warp % WarpsM)*WarpRows, wj = (warp / WarpsM)*WarpCols;

    const std::ptrdiff_t i0 = std::ptrdiff_t(blockIdx.x)*BlockRows;
    const std::ptrdiff_t j0 = std::ptrdiff_t(blockIdx.y)*BlockCols;
    //  rows and columns of this block that are inside C
    const int mb = m - i0 < BlockRows ? int(m - i0) : BlockRows;
    const int nb = n - j0 < BlockCols ? int(n - j0) : BlockCols;

    A += i0;
    B += j0*ldB;

    wmma::fragment<wmma::accumulator, TileM, TileN, TileK, double>
        ab[TilesM][TilesN];
    for (int r = 0; r < TilesM; ++r) {
        for (int c = 0; c < TilesN; ++c) {
            wmma::fill_fragment(ab[r][c], 0.0);
        }
    }

    if (alpha != 0) {
        for (std::ptrdiff_t l0 = 0; l0 < k; l0 += Panel) {
            const int kb = k - l0 < Panel ? int(k - l0) : Panel;

            //  copy the panels; thread tid copies the elements tid,
            //  tid + NumThreads, ... of a panel (numbered column by column)
            for (int q = 0; q < BlockRows*Panel/NumThreads; ++q) {
                int e = tid + q*NumThreads, i = e % BlockRows, l = e / BlockRows;
                sA[i + l*LdA] = (i < mb && l < kb) ? A[i + l*ldA] : 0;
            }
            for (int q = 0; q < Panel*BlockCols/NumThreads; ++q) {
                int e = tid + q*NumThreads, l = e % Panel, j = e / Panel;
                sB[l + j*LdB] = (j < nb && l < kb) ? B[l + j*ldB] : 0;
            }
            __syncthreads();

            #pragma unroll
            for (int s = 0; s < Panel/TileK; ++s) {
                wmma::fragment<wmma::matrix_a, TileM, TileN, TileK, double,
                               wmma::col_major> a[TilesM];
                wmma::fragment<wmma::matrix_b, TileM, TileN, TileK, double,
                               wmma::col_major> b[TilesN];
                for (int r = 0; r < TilesM; ++r) {
                    wmma::load_matrix_sync(a[r],
                        sA + wi + r*TileM + s*TileK*LdA, LdA);
                }
                for (int c = 0; c < TilesN; ++c) {
                    wmma::load_matrix_sync(b[c],
                        sB + s*TileK + (wj + c*TileN)*LdB, LdB);
                }
                for (int r = 0; r < TilesM; ++r) {
                    for (int c = 0; c < TilesN; ++c) {
                        wmma::mma_sync(ab[r][c], a[r], b[c], ab[r][c]);
                    }
                }
            }
            __syncthreads();

            A += std::ptrdiff_t(Panel)*ldA;
            B += Panel;
        }
    }

    //  C = beta*C + alpha*AB for the elements inside of C (beta == 0: C is
    //  not read, it may contain NaN); every warp stores its blocks into its
    //  part of buf (the panels are no longer used)
    double *sC = buf + warp*WarpRows*WarpCols;
    for (int r = 0; r < TilesM; ++r) {
        for (int c = 0; c < TilesN; ++c) {
            wmma::store_matrix_sync(sC + r*TileM + c*TileN*WarpRows,
                                    ab[r][c], WarpRows, wmma::mem_col_major);
        }
    }
    __syncwarp();
    for (int e = lane; e < WarpRows*WarpCols; e += 32) {
        int i = wi + e % WarpRows, j = wj + e / WarpRows;
        if (i < mb && j < nb) {
            double &cij = C[i0 + i + (j0 + j)*ldC];
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
        throw std::runtime_error("cULMblas::step08::gemm: needs a GPU with "
                                 "FP64 Tensor Cores (compute capability "
                                 ">= 8.0)");
    }

    std::ptrdiff_t numBlocksM = ceildiv(m, BlockRows);
    std::ptrdiff_t numBlocksN = ceildiv(n, BlockCols);

    if (numBlocksM > std::numeric_limits<int>::max() || numBlocksN > 65535) {
        throw std::invalid_argument("cULMblas::step08::gemm: m or n too "
                                    "large for the grid of this kernel");
    }
    //  offsets within a panel (up to Panel columns of A, BlockCols columns
    //  of B) are int
    constexpr std::ptrdiff_t intMax = std::numeric_limits<int>::max();
    if (ldA > intMax/Panel || ldB > intMax/BlockCols) {
        throw std::invalid_argument("cULMblas::step08::gemm: ldA or ldB too "
                                    "large for this kernel");
    }

    dim3 block(32, NumWarps);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    dgemm<<<grid, block, 0, stream>>>(m, n, k, alpha, A, int(ldA), B,
                                      int(ldB), beta, C, ldC);
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step08, cULMblas
