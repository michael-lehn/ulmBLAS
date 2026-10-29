//
//  Step 03 of the tutorial: reuse loaded elements of A and of B.
//
//  A thread computes an 8 x 4 block of C (ThreadRows x ThreadCols): the
//  rows i0, i0+32, ..., i0+224 and the columns j0, ..., j0+3.  For every l
//  it loads 8 elements of A and 4 of B into registers and does all 32
//  multiply-adds with them.  The rows are 32 apart, so that for every one
//  of the 8 loads of A the 32 threads of a warp still read 32 consecutive
//  elements of a column (coalesced), and all threads of a warp read the
//  same elements of B (broadcast).  A block of 32 x 8 threads covers
//  256 x 32 elements of C.
//
//  Rows and columns of a thread's block outside of C are loaded as 0 and
//  not stored, so the loop of multiply-adds needs no case distinctions.
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step03/gemm.h>

#include <limits>
#include <stdexcept>

namespace cULMblas { namespace step03 {

namespace {

//  ceil(a/b) for a >= 0, b > 0, without the overflow of (a + b - 1)/b
std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

__global__ void __launch_bounds__(ThreadsX*ThreadsY)
dgemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
      double alpha,
      const double *A, std::ptrdiff_t ldA,
      const double *B, std::ptrdiff_t ldB,
      double beta,
      double *C, std::ptrdiff_t ldC)
{
    std::ptrdiff_t i0 = std::ptrdiff_t(blockIdx.x)*BlockRows + threadIdx.x;
    std::ptrdiff_t j0 = std::ptrdiff_t(blockIdx.y)*BlockCols
                      + threadIdx.y*ThreadCols;

    //  which of the rows i0 + r*ThreadsX and columns j0 + c are inside C
    bool row[ThreadRows], col[ThreadCols];
    for (int r = 0; r < ThreadRows; ++r) {
        row[r] = i0 + r*ThreadsX < m;
    }
    for (int c = 0; c < ThreadCols; ++c) {
        col[c] = j0 + c < n;
    }
    if (!row[0] || !col[0]) {
        return;
    }

    double ab[ThreadRows][ThreadCols] = {};
    if (alpha != 0) {
        for (std::ptrdiff_t l = 0; l < k; ++l) {
            double a[ThreadRows], b[ThreadCols];
            for (int r = 0; r < ThreadRows; ++r) {
                a[r] = row[r] ? A[i0 + r*ThreadsX + l*ldA] : 0;
            }
            for (int c = 0; c < ThreadCols; ++c) {
                b[c] = col[c] ? B[l + (j0 + c)*ldB] : 0;
            }
            for (int r = 0; r < ThreadRows; ++r) {
                for (int c = 0; c < ThreadCols; ++c) {
                    ab[r][c] += a[r]*b[c];
                }
            }
        }
    }

    //  beta == 0: C is not read, it may contain NaN
    for (int r = 0; r < ThreadRows; ++r) {
        for (int c = 0; c < ThreadCols; ++c) {
            if (row[r] && col[c]) {
                double &cij = C[i0 + r*ThreadsX + (j0 + c)*ldC];
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
        throw std::invalid_argument("cULMblas::step03::gemm: m or n too "
                                    "large for the grid of this kernel");
    }

    dim3 block(ThreadsX, ThreadsY);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    dgemm<<<grid, block, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB,
                                      beta, C, ldC);
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step03, cULMblas
