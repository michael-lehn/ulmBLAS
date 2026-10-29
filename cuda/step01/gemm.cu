//
//  Step 01 of the tutorial: the naive GEMM kernel.
//
//  C is partitioned into blocks of 32 x 32 elements, one block of threads
//  per block of C, and each thread computes one element:
//
//      C(i,j) <- beta*C(i,j) + alpha * sum_l A(i,l)*B(l,j)
//
//  Threads with consecutive threadIdx.x have consecutive row indices i, so
//  a warp reads 32 consecutive elements of a column of A and writes 32
//  consecutive elements of a column of C (coalesced); all threads of a warp
//  read the same element of B.
//
//  This is the kernel of session 32 of my HPC I course.  Indices are
//  computed in 64 bits, for m == 0 or n == 0 nothing is launched (a grid of
//  size 0 is a launch error), alpha == 0 does not read A and B, C is read
//  and written once, and the launch is checked.
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step01/gemm.h>

#include <limits>
#include <stdexcept>

namespace cULMblas { namespace step01 {

namespace {

//  ceil(a/b) for a >= 0, b > 0, without the overflow of (a + b - 1)/b
std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

//  __launch_bounds__: a block has 32 x 32 = 1024 threads, the maximum.  The
//  compiler then limits the registers per thread so that a block fits on a
//  multiprocessor; otherwise the launch could fail with "too many resources
//  requested".
__global__ void __launch_bounds__(BlockDim*BlockDim)
dgemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
      double alpha,
      const double *A, std::ptrdiff_t ldA,
      const double *B, std::ptrdiff_t ldB,
      double beta,
      double *C, std::ptrdiff_t ldC)
{
    std::ptrdiff_t i = std::ptrdiff_t(blockIdx.x)*blockDim.x + threadIdx.x;
    std::ptrdiff_t j = std::ptrdiff_t(blockIdx.y)*blockDim.y + threadIdx.y;

    //  threads of the blocks at the bottom and the right of C idle
    if (i >= m || j >= n) {
        return;
    }

    double ab = 0;
    if (alpha != 0) {
        for (std::ptrdiff_t l = 0; l < k; ++l) {
            ab += A[i + l*ldA] * B[l + j*ldB];
        }
    }

    //  beta == 0: C is not read, it may contain NaN
    double &c = C[i + j*ldC];
    c = (beta == 0) ? alpha*ab : beta*c + alpha*ab;
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

    //  nothing to do (no grid of size 0: that is not a valid launch)
    if (m == 0 || n == 0 || ((alpha == 0 || k == 0) && beta == 1)) {
        return;
    }

    std::ptrdiff_t numBlocksM = ceildiv(m, BlockDim);
    std::ptrdiff_t numBlocksN = ceildiv(n, BlockDim);

    //  limits of the grid: gridDim.x <= 2^31 - 1, gridDim.y <= 65535
    if (numBlocksM > std::numeric_limits<int>::max() || numBlocksN > 65535) {
        throw std::invalid_argument("cULMblas::step01::gemm: m or n too "
                                    "large for the grid of this kernel");
    }

    dim3 block(BlockDim, BlockDim);
    dim3 grid(static_cast<unsigned>(numBlocksM),
              static_cast<unsigned>(numBlocksN));

    dgemm<<<grid, block, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB,
                                      beta, C, ldC);
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} } // namespace step01, cULMblas
