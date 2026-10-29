//
//  Step 02 of the tutorial: reuse a loaded element of A.
//
//  In step 01 every multiply-add needs two loads, A(i,l) and B(l,j).  Here
//  a thread computes ThreadCols (4) consecutive elements of a row of C,
//
//      C(i,j0), ..., C(i,j0+3),
//
//  so A(i,l) is loaded once and used for 4 multiply-adds; the 4 partial
//  sums stay in registers.  A block of 32 x 8 threads still covers 32 x 32
//  elements of C.  The accesses keep the pattern of step 01: the 32 threads
//  of a warp have consecutive i (coalesced loads of A and stores of C) and
//  the same j0 (each load of B is a broadcast).
//
//  Only the boundary check changes: a thread at the right edge of C may
//  have fewer than 4 columns of C left (nc < 4).
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step02/gemm.h>

#include <limits>
#include <stdexcept>

namespace cULMblas { namespace step02 {

namespace {

//  ceil(a/b) for a >= 0, b > 0, without the overflow of (a + b - 1)/b
std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

__global__ void __launch_bounds__(BlockRows*ThreadsY)
dgemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
      double alpha,
      const double *A, std::ptrdiff_t ldA,
      const double *B, std::ptrdiff_t ldB,
      double beta,
      double *C, std::ptrdiff_t ldC)
{
    std::ptrdiff_t i  = std::ptrdiff_t(blockIdx.x)*BlockRows + threadIdx.x;
    std::ptrdiff_t j0 = std::ptrdiff_t(blockIdx.y)*BlockCols
                      + threadIdx.y*ThreadCols;

    //  number of the columns j0, ..., j0+ThreadCols-1 that are inside C
    std::ptrdiff_t nc = n - j0 < ThreadCols ? n - j0 : ThreadCols;

    if (i >= m || nc <= 0) {
        return;
    }

    double ab[ThreadCols] = {};
    if (alpha != 0) {
        for (std::ptrdiff_t l = 0; l < k; ++l) {
            double a = A[i + l*ldA];
            for (int c = 0; c < ThreadCols; ++c) {
                if (c < nc) {
                    ab[c] += a * B[l + (j0 + c)*ldB];
                }
            }
        }
    }

    //  beta == 0: C is not read, it may contain NaN
    for (int c = 0; c < ThreadCols; ++c) {
        if (c < nc) {
            double &cij = C[i + (j0 + c)*ldC];
            cij = (beta == 0) ? alpha*ab[c] : beta*cij + alpha*ab[c];
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
        throw std::invalid_argument("cULMblas::step02::gemm: m or n too "
                                    "large for the grid of this kernel");
    }

    dim3 block(BlockRows, ThreadsY);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    dgemm<<<grid, block, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB,
                                      beta, C, ldC);
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step02, cULMblas
