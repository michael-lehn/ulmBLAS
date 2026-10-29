//
//  Step 09 (experimental knobs): as step 08, but the panels are copied with
//  cp.async into Stages buffers in shared memory, so that the copies of
//  the next Stages - 1 panels are on the way while the warps compute with
//  the current one.
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step09/gemm.h>

#include <cuda_pipeline.h>
#include <mma.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#ifndef CULMBLAS_STEP09_PAD
#define CULMBLAS_STEP09_PAD 4
#endif
#ifndef CULMBLAS_STEP09_VEC             // 1: 16-byte copies if A and B allow it
#define CULMBLAS_STEP09_VEC 1
#endif

namespace cULMblas { namespace step09 {

namespace {

constexpr int NumWarps   = WarpsM*WarpsN;
constexpr int NumThreads = 32*NumWarps;
constexpr int Pad        = CULMBLAS_STEP09_PAD;
constexpr int LdA        = BlockRows + Pad;             // of sA
constexpr int LdB        = Panel + Pad;                 // of sB
constexpr int SizeA      = LdA*Panel;                   // one panel, doubles
constexpr int SizeB      = LdB*BlockCols;
constexpr int SizeStage  = SizeA + SizeB;
constexpr int SizeC      = NumWarps*WarpRows*WarpCols;
constexpr int SizeBuf    = Stages*SizeStage > SizeC ? Stages*SizeStage : SizeC;

static_assert(LdA % 2 == 0 && LdB % 2 == 0 && SizeA % 4 == 0
              && SizeStage % 4 == 0,
              "alignment of the fragments in shared memory");
static_assert(BlockRows*Panel % NumThreads == 0,
              "the threads must copy the panel of A in equal parts");
static_assert(Panel*BlockCols % NumThreads == 0,
              "the threads must copy the panel of B in equal parts");
static_assert(Panel % TileK == 0, "a panel is a multiple of 4 values of l");
static_assert(Stages >= 2, "at least two buffers");

__host__ __device__ std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

template <bool Vec>
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

    extern __shared__ __align__(32) double buf[];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid = warp*32 + lane;
    const int wi = (warp % WarpsM)*WarpRows, wj = (warp / WarpsM)*WarpCols;

    const std::ptrdiff_t i0 = std::ptrdiff_t(blockIdx.x)*BlockRows;
    const std::ptrdiff_t j0 = std::ptrdiff_t(blockIdx.y)*BlockCols;
    const int mb = m - i0 < BlockRows ? int(m - i0) : BlockRows;
    const int nb = n - j0 < BlockCols ? int(n - j0) : BlockCols;

    A += i0;
    B += j0*ldB;

    const std::ptrdiff_t numPanels = alpha != 0 ? ceildiv(k, Panel) : 0;

    //  start the copies of panel t into buffer t % Stages (nothing for
    //  t >= numPanels); one commit per call, also if nothing was copied
    auto copyPanel = [&](std::ptrdiff_t t) {
        if (t < numPanels) {
            double *sA = buf + (t % Stages)*SizeStage, *sB = sA + SizeA;
            const std::ptrdiff_t l0 = t*Panel;
            const int kb = k - l0 < Panel ? int(k - l0) : Panel;
            const double *At = A + l0*ldA, *Bt = B + l0;
            if (Vec) {
                //  pairs of consecutive elements of a column, 16 bytes
                for (int q = 0; q < BlockRows*Panel/(2*NumThreads); ++q) {
                    int e = 2*(tid + q*NumThreads), i = e % BlockRows,
                        l = e / BlockRows;
                    if (i + 1 < mb && l < kb) {
                        __pipeline_memcpy_async(&sA[i + l*LdA],
                                                &At[i + l*ldA], 16);
                    } else {
                        sA[i + l*LdA] = (i < mb && l < kb) ? At[i + l*ldA] : 0;
                        sA[i + 1 + l*LdA] = 0;
                    }
                }
                for (int q = 0; q < Panel*BlockCols/(2*NumThreads); ++q) {
                    int e = 2*(tid + q*NumThreads), l = e % Panel, j = e / Panel;
                    if (j < nb && l + 1 < kb) {
                        __pipeline_memcpy_async(&sB[l + j*LdB],
                                                &Bt[l + j*ldB], 16);
                    } else {
                        sB[l + j*LdB] = (j < nb && l < kb) ? Bt[l + j*ldB] : 0;
                        sB[l + 1 + j*LdB] = 0;
                    }
                }
            } else {
            for (int q = 0; q < BlockRows*Panel/NumThreads; ++q) {
                int e = tid + q*NumThreads, i = e % BlockRows, l = e / BlockRows;
                if (i < mb && l < kb) {
                    __pipeline_memcpy_async(&sA[i + l*LdA], &At[i + l*ldA],
                                            sizeof(double));
                } else {
                    sA[i + l*LdA] = 0;
                }
            }
            for (int q = 0; q < Panel*BlockCols/NumThreads; ++q) {
                int e = tid + q*NumThreads, l = e % Panel, j = e / Panel;
                if (j < nb && l < kb) {
                    __pipeline_memcpy_async(&sB[l + j*LdB], &Bt[l + j*ldB],
                                            sizeof(double));
                } else {
                    sB[l + j*LdB] = 0;
                }
            }
            }
        }
        __pipeline_commit();
    };

    wmma::fragment<wmma::accumulator, TileM, TileN, TileK, double>
        ab[TilesM][TilesN];
    for (int r = 0; r < TilesM; ++r) {
        for (int c = 0; c < TilesN; ++c) {
            wmma::fill_fragment(ab[r][c], 0.0);
        }
    }

    for (int t = 0; t < Stages - 1; ++t) {
        copyPanel(t);
    }
    for (std::ptrdiff_t t = 0; t < numPanels; ++t) {
        //  my copies of panel t are done, then everybody's are; and
        //  everybody is done with panel t - 1, whose buffer is refilled now
        __pipeline_wait_prior(Stages - 2);
        __syncthreads();
        copyPanel(t + Stages - 1);

        const double *sA = buf + (t % Stages)*SizeStage, *sB = sA + SizeA;
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
    }
    __pipeline_wait_prior(0);
    __syncthreads();

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
        throw std::runtime_error("cULMblas::step09::gemm: needs a GPU with "
                                 "FP64 Tensor Cores (compute capability "
                                 ">= 8.0)");
    }

    std::ptrdiff_t numBlocksM = ceildiv(m, BlockRows);
    std::ptrdiff_t numBlocksN = ceildiv(n, BlockCols);

    if (numBlocksM > std::numeric_limits<int>::max() || numBlocksN > 65535) {
        throw std::invalid_argument("cULMblas::step09::gemm: m or n too "
                                    "large for the grid of this kernel");
    }
    constexpr std::ptrdiff_t intMax = std::numeric_limits<int>::max();
    if (ldA > intMax/Panel || ldB > intMax/BlockCols) {
        throw std::invalid_argument("cULMblas::step09::gemm: ldA or ldB too "
                                    "large for this kernel");
    }

    //  more than 48 KB of shared memory per block must be asked for
    constexpr int shmem = SizeBuf*sizeof(double);
    static const bool shmemSet = [] {
        CULMBLAS_CHECK_CUDA(cudaFuncSetAttribute(
            dgemm<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        CULMBLAS_CHECK_CUDA(cudaFuncSetAttribute(
            dgemm<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        return true;
    }();
    (void)shmemSet;

    //  16-byte copies: A and B 16-byte aligned, leading dimensions even
    const bool vec = CULMBLAS_STEP09_VEC
                  && reinterpret_cast<std::uintptr_t>(A) % 16 == 0
                  && reinterpret_cast<std::uintptr_t>(B) % 16 == 0
                  && ldA % 2 == 0 && ldB % 2 == 0;

    dim3 block(32, NumWarps);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    if (vec) {
        dgemm<true><<<grid, block, shmem, stream>>>(m, n, k, alpha, A,
                                                    int(ldA), B, int(ldB),
                                                    beta, C, ldC);
    } else {
        dgemm<false><<<grid, block, shmem, stream>>>(m, n, k, alpha, A,
                                                     int(ldA), B, int(ldB),
                                                     beta, C, ldC);
    }
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step09, cULMblas
