//
//  Step 05 (experimental knobs, not final): panels of A and B in shared
//  memory.
//
//  A block of ThreadsX x ThreadsY threads computes BlockRows x BlockCols
//  elements of C.  The loop over l goes in panels of Panel values of l:
//  the threads of the block copy the BlockRows x Panel panel of A and the
//  Panel x BlockCols panel of B into shared memory (each element once),
//  wait for each other, and then every thread computes its ThreadRows x
//  ThreadCols elements of C from shared memory.
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step05/gemm.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

#ifndef CULMBLAS_STEP05_COL_STRIDE      // 1: neighboring columns, 0: ThreadsY apart
#define CULMBLAS_STEP05_COL_STRIDE 1
#endif
#ifndef CULMBLAS_STEP05_INDEX           // 0: std::ptrdiff_t, 1: int within a panel
#define CULMBLAS_STEP05_INDEX 1
#endif
#ifndef CULMBLAS_STEP05_MIN_BLOCKS
#define CULMBLAS_STEP05_MIN_BLOCKS 2
#endif
#ifndef CULMBLAS_STEP05_PAD
#define CULMBLAS_STEP05_PAD 1
#endif

namespace cULMblas { namespace step05 {

namespace {

constexpr int NumThreads = ThreadsX*ThreadsY;
constexpr int Pad = CULMBLAS_STEP05_PAD;

static_assert(BlockRows*Panel % NumThreads == 0, "panel of A");
static_assert(Panel*BlockCols % NumThreads == 0, "panel of B");

#if CULMBLAS_STEP05_INDEX
using Index = int;
#else
using Index = std::ptrdiff_t;
#endif

std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

__device__ int
colOf(int c)
{
#if CULMBLAS_STEP05_COL_STRIDE
    return threadIdx.y*ThreadCols + c;
#else
    return threadIdx.y + c*ThreadsY;
#endif
}

__global__ void
__launch_bounds__(NumThreads, CULMBLAS_STEP05_MIN_BLOCKS)
dgemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
      double alpha,
      const double *A, std::ptrdiff_t ldA_,
      const double *B, std::ptrdiff_t ldB_,
      double beta,
      double *C, std::ptrdiff_t ldC)
{
    __shared__ double sA[Panel][BlockRows + Pad];   // sA[l][i]: A(i,l)
    __shared__ double sB[BlockCols][Panel + Pad];   // sB[j][l]: B(l,j)

    const int tid = threadIdx.y*ThreadsX + threadIdx.x;
    const std::ptrdiff_t i0 = std::ptrdiff_t(blockIdx.x)*BlockRows;
    const std::ptrdiff_t j0 = std::ptrdiff_t(blockIdx.y)*BlockCols;
    //  rows and columns of the block inside C
    const Index mb = Index(m - i0 < BlockRows ? m - i0 : BlockRows);
    const Index nb = Index(n - j0 < BlockCols ? n - j0 : BlockCols);
    const Index ldA = Index(ldA_), ldB = Index(ldB_);

    A += i0;
    B += j0*ldB_;

    double ab[ThreadRows][ThreadCols] = {};
    if (alpha != 0) {
        for (std::ptrdiff_t l0 = 0; l0 < k; l0 += Panel) {
            const Index kb = Index(k - l0 < Panel ? k - l0 : Panel);
            for (int q = 0; q < BlockRows*Panel/NumThreads; ++q) {
                Index e = tid + q*NumThreads, i = e % BlockRows, l = e / BlockRows;
                sA[l][i] = (i < mb && l < kb) ? A[i + l*ldA] : 0;
            }
            for (int q = 0; q < Panel*BlockCols/NumThreads; ++q) {
                Index e = tid + q*NumThreads, l = e % Panel, j = e / Panel;
                sB[j][l] = (j < nb && l < kb) ? B[l + j*ldB] : 0;
            }
            __syncthreads();
            #pragma unroll
            for (int l = 0; l < Panel; ++l) {
                double a[ThreadRows], b[ThreadCols];
                for (int r = 0; r < ThreadRows; ++r) {
                    a[r] = sA[l][threadIdx.x + r*ThreadsX];
                }
                for (int c = 0; c < ThreadCols; ++c) {
                    b[c] = sB[colOf(c)][l];
                }
                for (int r = 0; r < ThreadRows; ++r) {
                    for (int c = 0; c < ThreadCols; ++c) {
                        ab[r][c] += a[r]*b[c];
                    }
                }
            }
            __syncthreads();
            A += std::ptrdiff_t(Panel)*ldA_;
            B += Panel;
        }
    }

    for (int r = 0; r < ThreadRows; ++r) {
        for (int c = 0; c < ThreadCols; ++c) {
            Index i = threadIdx.x + r*ThreadsX, j = colOf(c);
            if (i < mb && j < nb) {
                double &cij = C[i0 + i + (j0 + j)*ldC];
                cij = (beta == 0) ? alpha*ab[r][c]
                                  : beta*cij + alpha*ab[r][c];
            }
        }
    }
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

    std::ptrdiff_t numBlocksM = ceildiv(m, BlockRows);
    std::ptrdiff_t numBlocksN = ceildiv(n, BlockCols);

    if (numBlocksM > std::numeric_limits<int>::max() || numBlocksN > 65535) {
        throw std::invalid_argument("cULMblas::step05::gemm: m or n too "
                                    "large for the grid of this kernel");
    }
#if CULMBLAS_STEP05_INDEX
    //  offsets within a panel are ints
    if (std::max(ldA, ldB)*(Panel + 1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("cULMblas::step05::gemm: leading "
                                    "dimension too large");
    }
#endif

    dim3 block(ThreadsX, ThreadsY);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    dgemm<<<grid, block, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB,
                                      beta, C, ldC);
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step05, cULMblas
