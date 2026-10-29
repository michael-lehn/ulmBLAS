//
//  Step 04 of the tutorial: several values of l per pass.
//
//  As in step 03 a thread computes a block of C (ThreadRows x ThreadCols,
//  default 4 x 4) with rows 32 apart and neighboring columns.  The loop
//  over l is done in passes of Pass values of l (default 4): a pass first
//  loads the Pass columns of A and rows of B it needs (Pass x (ThreadRows +
//  ThreadCols) independent loads) and then does all their multiply-adds.
//  So a warp has all loads of a pass on the way at the same time and waits
//  once per pass, not once per value of l.
//
//  Only whole passes are done this way, without a check l < k inside; the
//  last k % Pass values of l are done one by one as in step 03.
//
//  A block of 32 x 8 threads covers 128 x 32 elements of C.  Rows and
//  columns of a thread's block outside of C are loaded as 0 and not stored.
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step04/gemm.h>

#include <limits>
#include <stdexcept>

namespace cULMblas { namespace step04 {

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
        std::ptrdiff_t l = 0;
        for (; l + Pass <= k; l += Pass) {
            double a[Pass][ThreadRows], b[Pass][ThreadCols];
            for (int p = 0; p < Pass; ++p) {
                for (int r = 0; r < ThreadRows; ++r) {
                    a[p][r] = row[r] ? A[i0 + r*ThreadsX + (l + p)*ldA] : 0;
                }
                for (int c = 0; c < ThreadCols; ++c) {
                    b[p][c] = col[c] ? B[l + p + (j0 + c)*ldB] : 0;
                }
            }
            for (int p = 0; p < Pass; ++p) {
                for (int r = 0; r < ThreadRows; ++r) {
                    for (int c = 0; c < ThreadCols; ++c) {
                        ab[r][c] += a[p][r]*b[p][c];
                    }
                }
            }
        }
        //  the last k % Pass values of l, one by one as in step 03
        for (; l < k; ++l) {
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
        throw std::invalid_argument("cULMblas::step04::gemm: m or n too "
                                    "large for the grid of this kernel");
    }

    dim3 block(ThreadsX, ThreadsY);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    dgemm<<<grid, block, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB,
                                      beta, C, ldC);
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step04, cULMblas
