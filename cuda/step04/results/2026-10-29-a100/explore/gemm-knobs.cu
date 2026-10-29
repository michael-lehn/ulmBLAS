//
//  Step 04 (experimental knobs, not final).
//
#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step04/gemm.h>

#include <limits>
#include <stdexcept>

#ifndef CULMBLAS_STEP04_UNROLL
#define CULMBLAS_STEP04_UNROLL 1
#endif
#ifndef CULMBLAS_STEP04_PREFETCH
#define CULMBLAS_STEP04_PREFETCH 1
#endif
#ifndef CULMBLAS_STEP04_MIN_BLOCKS
#define CULMBLAS_STEP04_MIN_BLOCKS 1
#endif

namespace cULMblas { namespace step04 {

namespace {

constexpr int Unroll = CULMBLAS_STEP04_UNROLL;

std::ptrdiff_t
ceildiv(std::ptrdiff_t a, std::ptrdiff_t b)
{
    return a/b + (a%b != 0);
}

__global__ void
__launch_bounds__(ThreadsX*ThreadsY, CULMBLAS_STEP04_MIN_BLOCKS)
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

    auto load = [&](std::ptrdiff_t l, double (&a)[Unroll][ThreadRows],
                    double (&b)[Unroll][ThreadCols])
    {
        for (int u = 0; u < Unroll; ++u) {
            bool inK = l + u < k;
            for (int r = 0; r < ThreadRows; ++r) {
                a[u][r] = row[r] && inK ? A[i0 + r*ThreadsX + (l + u)*ldA]
                                        : 0;
            }
            for (int c = 0; c < ThreadCols; ++c) {
                b[u][c] = col[c] && inK ? B[l + u + (j0 + c)*ldB] : 0;
            }
        }
    };

    double ab[ThreadRows][ThreadCols] = {};
    if (alpha != 0) {
        double a[Unroll][ThreadRows], b[Unroll][ThreadCols];
#if CULMBLAS_STEP04_PREFETCH == 1
        load(0, a, b);
#endif
#if CULMBLAS_STEP04_PREFETCH == 2
        #pragma unroll Unroll
        for (std::ptrdiff_t l = 0; l < k; ++l) {
            double a1[ThreadRows], b1[ThreadCols];
            for (int r = 0; r < ThreadRows; ++r) {
                a1[r] = row[r] ? A[i0 + r*ThreadsX + l*ldA] : 0;
            }
            for (int c = 0; c < ThreadCols; ++c) {
                b1[c] = col[c] ? B[l + (j0 + c)*ldB] : 0;
            }
            for (int r = 0; r < ThreadRows; ++r) {
                for (int c = 0; c < ThreadCols; ++c) {
                    ab[r][c] += a1[r]*b1[c];
                }
            }
        }
        if (false)
#endif
#if CULMBLAS_STEP04_PREFETCH == 3
        std::ptrdiff_t l = 0;
        for (; l + Unroll <= k; l += Unroll) {
            double a3[Unroll][ThreadRows], b3[Unroll][ThreadCols];
            for (int u = 0; u < Unroll; ++u) {
                for (int r = 0; r < ThreadRows; ++r) {
                    a3[u][r] = row[r] ? A[i0 + r*ThreadsX + (l + u)*ldA] : 0;
                }
                for (int c = 0; c < ThreadCols; ++c) {
                    b3[u][c] = col[c] ? B[l + u + (j0 + c)*ldB] : 0;
                }
            }
            for (int u = 0; u < Unroll; ++u) {
                for (int r = 0; r < ThreadRows; ++r) {
                    for (int c = 0; c < ThreadCols; ++c) {
                        ab[r][c] += a3[u][r]*b3[u][c];
                    }
                }
            }
        }
        for (; l < k; ++l) {
            for (int r = 0; r < ThreadRows; ++r) {
                for (int c = 0; c < ThreadCols; ++c) {
                    ab[r][c] += (row[r] ? A[i0 + r*ThreadsX + l*ldA] : 0)
                              * (col[c] ? B[l + (j0 + c)*ldB] : 0);
                }
            }
        }
        if (false)
#endif
        for (std::ptrdiff_t l = 0; l < k; l += Unroll) {
#if CULMBLAS_STEP04_PREFETCH == 1
            double an[Unroll][ThreadRows], bn[Unroll][ThreadCols];
            load(l + Unroll, an, bn);
#else
            load(l, a, b);
#endif
            for (int u = 0; u < Unroll; ++u) {
                for (int r = 0; r < ThreadRows; ++r) {
                    for (int c = 0; c < ThreadCols; ++c) {
                        ab[r][c] += a[u][r]*b[u][c];
                    }
                }
            }
#if CULMBLAS_STEP04_PREFETCH == 1
            for (int u = 0; u < Unroll; ++u) {
                for (int r = 0; r < ThreadRows; ++r) {
                    a[u][r] = an[u][r];
                }
                for (int c = 0; c < ThreadCols; ++c) {
                    b[u][c] = bn[u][c];
                }
            }
#endif
        }
    }

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
