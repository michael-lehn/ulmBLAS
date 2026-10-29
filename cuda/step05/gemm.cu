//
//  Step 05 of the tutorial: panels of A and B in shared memory.
//
//  A block of 16 x 8 threads (ThreadsX x ThreadsY) computes 64 x 32
//  elements of C (BlockRows x BlockCols); every thread computes a 4 x 4
//  block of it as in step 04: the rows threadIdx.x + 16*r and the
//  neighboring columns 4*threadIdx.y + c.
//
//  The loop over l goes in panels of 16 values of l (Panel).  For every
//  panel the threads of the block first copy the 64 x 16 panel of A and
//  the 16 x 32 panel of B into shared memory, each element of A and B once
//  and with consecutive threads reading consecutive elements of a column,
//  then wait for each other (__syncthreads), and then every thread does
//  its 16 x 16 multiply-adds of the panel with operands from shared
//  memory.  Before the next panel is copied, all threads must be done with
//  this one (the second __syncthreads).
//
//  Elements outside of A, B (rows, columns or values of l beyond m, n, k)
//  are stored as 0 in shared memory, so the multiply-adds need no case
//  distinctions; elements of C outside of C are not stored.
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step05/gemm.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cULMblas { namespace step05 {

namespace {

constexpr int NumThreads = ThreadsX*ThreadsY;

static_assert(BlockRows*Panel % NumThreads == 0,
              "the threads must copy the panel of A in equal parts");
static_assert(Panel*BlockCols % NumThreads == 0,
              "the threads must copy the panel of B in equal parts");

//  ceil(a/b) for a >= 0, b > 0, without the overflow of (a + b - 1)/b
std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

//  Indices within a block of C and offsets within a panel are int (the
//  caller checks that they fit): with std::ptrdiff_t the kernel needs more
//  registers and is slower (see README.md).  As in step 04, the 1 (at
//  least one block per SM) lets the compiler use the registers it needs.
__global__ void __launch_bounds__(ThreadsX*ThreadsY, 1)
dgemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
      double alpha,
      const double *A, int ldA,
      const double *B, int ldB,
      double beta,
      double *C, std::ptrdiff_t ldC)
{
    __shared__ double sA[Panel][BlockRows];         // sA[l][i]: A(i,l)
    __shared__ double sB[BlockCols][Panel];         // sB[j][l]: B(l,j)

    const int tid = threadIdx.y*ThreadsX + threadIdx.x;
    const std::ptrdiff_t i0 = std::ptrdiff_t(blockIdx.x)*BlockRows;
    const std::ptrdiff_t j0 = std::ptrdiff_t(blockIdx.y)*BlockCols;

    //  rows and columns of this block that are inside C
    const int mb = m - i0 < BlockRows ? int(m - i0) : BlockRows;
    const int nb = n - j0 < BlockCols ? int(n - j0) : BlockCols;

    //  the panels of this block start at A(i0, l0) and B(l0, j0)
    A += i0;
    B += j0*ldB;

    double ab[ThreadRows][ThreadCols] = {};
    if (alpha != 0) {
        for (std::ptrdiff_t l0 = 0; l0 < k; l0 += Panel) {
            const int kb = k - l0 < Panel ? int(k - l0) : Panel;

            //  copy the panels; thread tid copies the elements tid,
            //  tid + NumThreads, ... of a panel (numbered column by column)
            for (int q = 0; q < BlockRows*Panel/NumThreads; ++q) {
                int e = tid + q*NumThreads, i = e % BlockRows, l = e / BlockRows;
                sA[l][i] = (i < mb && l < kb) ? A[i + l*ldA] : 0;
            }
            for (int q = 0; q < Panel*BlockCols/NumThreads; ++q) {
                int e = tid + q*NumThreads, l = e % Panel, j = e / Panel;
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
                    b[c] = sB[threadIdx.y*ThreadCols + c][l];
                }
                for (int r = 0; r < ThreadRows; ++r) {
                    for (int c = 0; c < ThreadCols; ++c) {
                        ab[r][c] += a[r]*b[c];
                    }
                }
            }
            __syncthreads();

            A += std::ptrdiff_t(Panel)*ldA;
            B += Panel;
        }
    }

    //  beta == 0: C is not read, it may contain NaN
    for (int r = 0; r < ThreadRows; ++r) {
        for (int c = 0; c < ThreadCols; ++c) {
            int i = threadIdx.x + r*ThreadsX, j = threadIdx.y*ThreadCols + c;
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
    //  offsets within a panel (up to Panel columns of A, BlockCols columns
    //  of B) are int
    constexpr std::ptrdiff_t intMax = std::numeric_limits<int>::max();
    if (ldA > intMax/Panel || ldB > intMax/BlockCols) {
        throw std::invalid_argument("cULMblas::step05::gemm: ldA or ldB too "
                                    "large for this kernel");
    }

    dim3 block(ThreadsX, ThreadsY);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    dgemm<<<grid, block, 0, stream>>>(m, n, k, alpha, A, int(ldA), B,
                                      int(ldB), beta, C, ldC);
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step05, cULMblas
