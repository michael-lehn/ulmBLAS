//
//  Step 10 of the tutorial: mma.sync with an own assignment of the
//  operands.
//
//  The Tensor Core operation for double, mma.sync.m8n8k4 (PTX, instead of
//  WMMA), with the 32 threads of a warp; lane = 4*g + tq (g = 0..7,
//  tq = 0..3):
//
//      the thread holds  A(g, tq), B(tq, g), C(g, 2*tq), C(g, 2*tq + 1)
//
//  of the 8 x 4 block of A, the 4 x 8 block of B and the 8 x 8 block of C
//  of the operation (PTX ISA, "Matrix Fragments for mma.m8n8k4 with
//  .f64").  Free to choose:
//
//  - which values of l of a panel go into which of its S = Panel/4
//    operations: operation s has l = 8*(s/2) + 2*tq + s%2, so a thread
//    needs pairs of neighboring values of l (neighbors in a column of B);
//  - which rows of the warp's 32 rows block r of 8 x 8 covers: row
//    wi + 16*(r/2) + 2*g + r%2, so a thread needs pairs of neighboring rows
//    (neighbors in a column of A);
//  - the columns: block c has the columns wj + TilesN*n + c (n = 0..7).
//
//  So a thread loads its operands with 16-byte loads (two doubles) from
//  the panels, half as many loads as in step 09, and stores its elements
//  of C directly from its registers.  The choice of the pairs and the
//  lengths of the columns of the panels (LdA = 64 + 2, LdB = 16 + 4) make
//  the 16-byte loads of every 8 threads hit different banks of shared
//  memory (see README.md).
//
//  The panels are copied as in step 09 (cp.async, 2 buffers).
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step10/gemm.h>

#include <cuda_pipeline.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cULMblas { namespace step10 {

namespace {

//  C(8 x 8) += A(8 x 4) B(4 x 8) for one warp, operands as above; the
//  thread gives its a and b and updates its c0, c1
__device__ inline void
mma884(double &c0, double &c1, double a, double b)
{
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                 "{%0, %1}, {%2}, {%3}, {%0, %1};\n"
                 : "+d"(c0), "+d"(c1) : "d"(a), "d"(b));
}

constexpr int NumWarps   = WarpsM*WarpsN;
constexpr int NumThreads = 32*NumWarps;
constexpr int LdA        = BlockRows + 2;               // of sA, see above
constexpr int LdB        = Panel + (TilesN == 2 ? 4 : 2);   // of sB
constexpr int S          = Panel/TileK;                 // operations per panel
constexpr int SizeA      = LdA*Panel;                   // one panel, doubles
constexpr int SizeB      = LdB*BlockCols;
constexpr int SizeStage  = SizeA + SizeB;
constexpr int SizeBuf    = Stages*SizeStage;

//  16-byte copies and loads: even offsets in shared memory
static_assert(LdA % 2 == 0 && LdB % 2 == 0 && SizeA % 2 == 0
              && SizeStage % 2 == 0,
              "alignment of the 16-byte accesses in shared memory");
static_assert(BlockRows % 2 == 0 && Panel % 2 == 0, "pairs of elements");
static_assert(TilesM % 2 == 0 && Panel % 8 == 0,
              "rows 16*(r/2) + 2*g + r%2, values of l 8*(s/2) + 2*tq + s%2");
//  the 16-byte loads of 8 threads (g = 0, 1; tq = 0..3) in different banks
//  (16 doubles): A at 2*g + 2*tq*LdA, B at 2*tq + g*TilesN*LdB (mod 16)
static_assert(LdA % 8 == 2 && TilesN*LdB % 16 == 8,
              "panels without bank conflicts for the 16-byte loads");
static_assert(BlockRows*Panel % NumThreads == 0,
              "the threads must copy the panel of A in equal parts");
static_assert(Panel*BlockCols % NumThreads == 0,
              "the threads must copy the panel of B in equal parts");
static_assert(Panel % TileK == 0, "a panel is a multiple of 4 values of l");
static_assert(Stages >= 2, "at least two buffers");

//  ceil(a/b) for a >= 0, b > 0, without the overflow of (a + b - 1)/b
__host__ __device__ std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

//  Vec: 16-byte copies of the panels; threadIdx.x: the thread in its warp
//  (0..31), threadIdx.y: the warp
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
    //  Stages buffers, each with a panel of A (sA(i,l) = sA[i + l*LdA]) and
    //  of B (sB(l,j) = sB[l + j*LdB])
    extern __shared__ __align__(16) double buf[];

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
                //  single elements, 8 bytes
                for (int q = 0; q < BlockRows*Panel/NumThreads; ++q) {
                    int e = tid + q*NumThreads, i = e % BlockRows,
                        l = e / BlockRows;
                    if (i < mb && l < kb) {
                        __pipeline_memcpy_async(&sA[i + l*LdA],
                                                &At[i + l*ldA], 8);
                    } else {
                        sA[i + l*LdA] = 0;
                    }
                }
                for (int q = 0; q < Panel*BlockCols/NumThreads; ++q) {
                    int e = tid + q*NumThreads, l = e % Panel, j = e / Panel;
                    if (j < nb && l < kb) {
                        __pipeline_memcpy_async(&sB[l + j*LdB],
                                                &Bt[l + j*ldB], 8);
                    } else {
                        sB[l + j*LdB] = 0;
                    }
                }
            }
        }
        __pipeline_commit();
    };

    const int g = lane / 4, tq = lane % 4;

    //  ab[r][c][h]: C(wi + 16*(r/2) + 2*g + r%2, wj + TilesN*(2*tq + h) + c)
    double ab[TilesM][TilesN][2] = {};

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

        //  a[r][s] = A(wi + 16*(r/2) + 2*g + r%2, l),
        //  b[c][s] = B(l, wj + TilesN*g + c),  l = 8*(s/2) + 2*tq + s%2
        double a[TilesM][S], b[TilesN][S];
        for (int s = 0; s < S; ++s) {
            const int l = 8*(s/2) + 2*tq + s%2;
            for (int r = 0; r < TilesM; r += 2) {
                double2 v = *reinterpret_cast<const double2 *>(
                    &sA[wi + 16*(r/2) + 2*g + l*LdA]);
                a[r][s] = v.x;
                a[r + 1][s] = v.y;
            }
        }
        for (int c = 0; c < TilesN; ++c) {
            for (int s = 0; s < S; s += 2) {
                double2 v = *reinterpret_cast<const double2 *>(
                    &sB[8*(s/2) + 2*tq + (wj + TilesN*g + c)*LdB]);
                b[c][s] = v.x;
                b[c][s + 1] = v.y;
            }
        }
        for (int s = 0; s < S; ++s) {
            for (int r = 0; r < TilesM; ++r) {
                for (int c = 0; c < TilesN; ++c) {
                    mma884(ab[r][c][0], ab[r][c][1], a[r][s], b[c][s]);
                }
            }
        }
    }

    //  C = beta*C + alpha*AB for the elements inside of C (beta == 0: C is
    //  not read, it may contain NaN), directly from the registers
    for (int r = 0; r < TilesM; ++r) {
        int i = wi + 16*(r/2) + 2*g + r%2;
        for (int c = 0; c < TilesN; ++c) {
            for (int h = 0; h < 2; ++h) {
                int j = wj + TilesN*(2*tq + h) + c;
                if (i < mb && j < nb) {
                    double &cij = C[i0 + i + (j0 + j)*ldC];
                    cij = (beta == 0) ? alpha*ab[r][c][h]
                                      : beta*cij + alpha*ab[r][c][h];
                }
            }
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
        throw std::runtime_error("cULMblas::step10::gemm: needs a GPU with "
                                 "FP64 Tensor Cores (compute capability "
                                 ">= 8.0)");
    }

    std::ptrdiff_t numBlocksM = ceildiv(m, BlockRows);
    std::ptrdiff_t numBlocksN = ceildiv(n, BlockCols);

    if (numBlocksM > std::numeric_limits<int>::max() || numBlocksN > 65535) {
        throw std::invalid_argument("cULMblas::step10::gemm: m or n too "
                                    "large for the grid of this kernel");
    }
    //  offsets within a panel (up to Panel columns of A, BlockCols columns
    //  of B) are int
    constexpr std::ptrdiff_t intMax = std::numeric_limits<int>::max();
    if (ldA > intMax/Panel || ldB > intMax/BlockCols) {
        throw std::invalid_argument("cULMblas::step10::gemm: ldA or ldB too "
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
    const bool vec = reinterpret_cast<std::uintptr_t>(A) % 16 == 0
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

} } // namespace step10, cULMblas
