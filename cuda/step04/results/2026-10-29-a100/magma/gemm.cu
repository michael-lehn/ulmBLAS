#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <xmagma/gemm.h>

namespace cULMblas { namespace xmagma {

namespace {

//  Shared: 0 = no shared memory, 1 = panels in shared memory,
//          2 = panels in shared memory, next panel prefetched in registers
//  ColStride: distance of a thread's columns (DY: like MAGMA, 1: like step 03)
template <int DX, int DY, int BM, int BN, int BK, int Shared, int ColStride,
          int MinBlocks, typename Idx>
__global__ void __launch_bounds__(DX*DY, MinBlocks)
dgemm(Idx m, Idx n, Idx k, double alpha,
      const double *A, Idx ldA, const double *B, Idx ldB,
      double beta, double *C, Idx ldC)
{
    constexpr int TM = BM/DX, TN = BN/DY, NT = DX*DY;
    constexpr int LA = BM*BK/NT, LB = BK*BN/NT;
    constexpr int SA = Shared ? BK*(BM+1) : 1, SB = Shared ? BN*(BK+1) : 1;
    __shared__ double sA[SA], sB[SB];     // sA(i,l) = sA[i + l*(BM+1)]

    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty*DX + tx;
    const Idx i0 = Idx(blockIdx.x)*BM;
    const Idx j0 = Idx(blockIdx.y)*BN;
    //  the thread's rows i0 + tx + t*DX and columns j0 + cj(t)
    auto cj = [&](int t) { return ColStride == 1 ? ty*TN + t : ty + t*DY; };

    bool row[TM], col[TN];
    for (int t = 0; t < TM; ++t) row[t] = i0 + tx + t*DX < m;
    for (int t = 0; t < TN; ++t) col[t] = j0 + cj(t) < n;

    double ab[TM][TN] = {};

    if constexpr (Shared != 0) {
        double rA[LA], rB[LB];
        auto load = [&](Idx l0) {
            for (int q = 0; q < LA; ++q) {
                int e = tid + q*NT, i = e % BM, l = e / BM;
                rA[q] = (i0 + i < m && l0 + l < k)
                      ? A[i0 + i + (l0 + l)*ldA] : 0;
            }
            for (int q = 0; q < LB; ++q) {
                int e = tid + q*NT, l = e % BK, j = e / BK;
                rB[q] = (j0 + j < n && l0 + l < k)
                      ? B[l0 + l + (j0 + j)*ldB] : 0;
            }
        };
        auto store = [&]() {
            for (int q = 0; q < LA; ++q) {
                int e = tid + q*NT;
                sA[e % BM + (e / BM)*(BM+1)] = rA[q];
            }
            for (int q = 0; q < LB; ++q) {
                int e = tid + q*NT;
                sB[e % BK + (e / BK)*(BK+1)] = rB[q];
            }
        };
        auto multiply = [&]() {
            #pragma unroll
            for (int l = 0; l < BK; ++l) {
                double a[TM], b[TN];
                for (int t = 0; t < TM; ++t) a[t] = sA[tx + t*DX + l*(BM+1)];
                for (int t = 0; t < TN; ++t) b[t] = sB[l + cj(t)*(BK+1)];
                for (int r = 0; r < TM; ++r)
                    for (int c = 0; c < TN; ++c)
                        ab[r][c] += a[r]*b[c];
            }
        };
        if constexpr (Shared == 2) {
            load(0);
            store();
            __syncthreads();
            for (Idx l0 = 0; l0 < k; l0 += BK) {
                bool more = l0 + BK < k;
                if (more) load(l0 + BK);
                multiply();
                __syncthreads();
                if (more) store();
                __syncthreads();
            }
        } else {
            for (Idx l0 = 0; l0 < k; l0 += BK) {
                load(l0);
                store();
                __syncthreads();
                multiply();
                __syncthreads();
            }
        }
    } else {
        auto pass = [&](Idx l) {
            double a[TM], b[TN];
            for (int t = 0; t < TM; ++t)
                a[t] = row[t] ? A[i0 + tx + t*DX + l*ldA] : 0;
            for (int t = 0; t < TN; ++t)
                b[t] = col[t] ? B[l + (j0 + cj(t))*ldB] : 0;
            for (int r = 0; r < TM; ++r)
                for (int c = 0; c < TN; ++c)
                    ab[r][c] += a[r]*b[c];
        };
        Idx l = 0;
        for (; l + BK <= k; l += BK) {
            //  all loads of the pass first, then the multiply-adds
            double a[BK][TM], b[BK][TN];
            for (int u = 0; u < BK; ++u) {
                for (int t = 0; t < TM; ++t)
                    a[u][t] = row[t] ? A[i0 + tx + t*DX + (l + u)*ldA] : 0;
                for (int t = 0; t < TN; ++t)
                    b[u][t] = col[t] ? B[l + u + (j0 + cj(t))*ldB] : 0;
            }
            for (int u = 0; u < BK; ++u)
                for (int r = 0; r < TM; ++r)
                    for (int c = 0; c < TN; ++c)
                        ab[r][c] += a[u][r]*b[u][c];
        }
        for (; l < k; ++l) pass(l);
    }

    for (int r = 0; r < TM; ++r) {
        for (int c = 0; c < TN; ++c) {
            if (row[r] && col[c]) {
                double &cij = C[i0 + tx + r*DX + (j0 + cj(c))*ldC];
                cij = (beta == 0) ? alpha*ab[r][c]
                                  : beta*cij + alpha*ab[r][c];
            }
        }
    }
}

template <int DX, int DY, int BM, int BN, int BK, int Shared, int ColStride,
          int MinBlocks = 1, typename Idx = std::ptrdiff_t>
void
launch(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k, double alpha,
       const double *A, std::ptrdiff_t ldA, const double *B, std::ptrdiff_t ldB,
       double beta, double *C, std::ptrdiff_t ldC, cudaStream_t stream)
{
    checkGemmArguments(m, n, k, A, ldA, B, ldB, C, ldC);
    if (m == 0 || n == 0 || ((alpha == 0 || k == 0) && beta == 1)) {
        return;
    }
    if (alpha == 0) k = 0;
    dim3 block(DX, DY);
    dim3 grid(unsigned((m + BM - 1)/BM), unsigned((n + BN - 1)/BN));
    dgemm<DX, DY, BM, BN, BK, Shared, ColStride, MinBlocks, Idx>
        <<<grid, block, 0, stream>>>
        (Idx(m), Idx(n), Idx(k), alpha, A, Idx(ldA), B, Idx(ldB), beta, C,
         Idx(ldC));
    CULMBLAS_CHECK_CUDA(cudaGetLastError());
}

} // namespace

#define ARGS std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,           \
             double alpha, const double *A, std::ptrdiff_t ldA,              \
             const double *B, std::ptrdiff_t ldB, double beta,               \
             double *C, std::ptrdiff_t ldC, cudaStream_t s
#define PASS m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, s

void gemm0(ARGS) { launch<16, 16, 64, 64, 16, 2, 16>(PASS); }
void gemm1(ARGS) { launch<16, 16, 64, 64, 16, 1, 16>(PASS); }
void gemm2(ARGS) { launch<16, 16, 64, 64, 16, 0, 16>(PASS); }
void gemm3(ARGS) { launch<16, 16, 64, 64,  4, 0, 16>(PASS); }
void gemm4(ARGS) { launch<32,  8, 128, 32, 4, 0,  1>(PASS); }
//  shared memory without prefetch (xm1) and ways to fewer registers
void gemm5(ARGS) { launch<16, 16, 64, 64, 16, 1, 16, 1, int>(PASS); }
void gemm6(ARGS) { launch<16, 16, 64, 64, 16, 1, 16, 3>(PASS); }
void gemm7(ARGS) { launch<16, 16, 64, 64, 16, 1, 16, 4>(PASS); }
void gemm8(ARGS) { launch<16, 16, 64, 64, 16, 1, 16, 3, int>(PASS); }
void gemm9(ARGS) { launch<16, 16, 64, 64,  8, 1, 16>(PASS); }
//  shared memory, step 04 layout: 32 x 8 threads, 4 x 4, neighboring columns
void gemm10(ARGS) { launch<32, 8, 128, 32, 16, 1, 1>(PASS); }
void gemm11(ARGS) { launch<32, 8, 128, 32, 16, 1, 16 / 2>(PASS); }
void gemm12(ARGS) { launch<16, 16, 64, 64, 16, 1, 1>(PASS); }
void gemm13(ARGS) { launch<16, 16, 64, 64, 16, 2, 16, 3, int>(PASS); }

} } // namespace xmagma, cULMblas
