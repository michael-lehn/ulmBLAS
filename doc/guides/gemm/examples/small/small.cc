// Small matrices: C <- beta C + alpha A B without packing.  Tiles of 8 x 4
// elements of C in registers (GCC vector extensions), A read in place (column
// major), B element by element. Compared with the blocked gemm of
// blocked/gemm.h and with ulmBLAS for small squares and a few skinny shapes.
#include "../blocked/gemm.h"
#include "../blocked/kernels.h"
#include "../common/util.h"
#include <cstring>
#include <ulmblas/ulmblas.h>
#include <vector>

#if defined(HAVE_SIMD_KERNEL)
typedef blocked::SimdKernel UK;
#else
typedef blocked::VecKernel UK;
#endif

// vectors of two (NEON) or four (AVX2) doubles
#if defined(__AVX2__)
typedef double vec __attribute__((vector_size(32)));
#else
typedef double vec __attribute__((vector_size(16)));
#endif
const long L = sizeof(vec) / sizeof(double), MV = 8 / L;
// the same for loads from any address of a double (unaligned, may alias)
typedef double uvec
    __attribute__((vector_size(sizeof(vec)), aligned(8), may_alias));

// an 8 x N tile of C: per l one column of A (MV vectors) times N elements
// of a row of B
template <long N>
void
tile(long k, double alpha, const double *A, long ldA, const double *B,
     long incRowB, long incColB, double beta, double *C, long ldC)
{
    vec c[N][MV] = {};
    for (long l = 0; l < k; ++l) {
	vec a[MV];
	for (long r = 0; r < MV; ++r) {
	    a[r] = *(const uvec *)&A[l * ldA + r * L];
	}
	for (long j = 0; j < N; ++j) {
	    const double b = B[l * incRowB + j * incColB];
	    for (long r = 0; r < MV; ++r) {
		c[j][r] += a[r] * b;
	    }
	}
    }
    double t[8 * N]; // the tile, column major
    std::memcpy(t, c, sizeof(t));
    for (long j = 0; j < N; ++j) {
	for (long i = 0; i < 8; ++i) {
	    double &x = C[i + j * ldC];
	    x = alpha * t[i + j * 8] + (beta == 0 ? 0 : beta * x);
	}
    }
}

// the elements of C outside the full tiles, one by one
void
rest(long m0, long m, long n0, long n, long k, double alpha, const double *A,
     long ldA, const double *B, long incRowB, long incColB, double beta,
     double *C, long ldC)
{
    for (long j = n0; j < n; ++j) {
	for (long i = m0; i < m; ++i) {
	    double s = 0;
	    for (long l = 0; l < k; ++l) {
		s += A[i + l * ldA] * B[l * incRowB + j * incColB];
	    }
	    double &x = C[i + j * ldC];
	    x = alpha * s + (beta == 0 ? 0 : beta * x);
	}
    }
}

// A and C column major (ldA, ldC), B with any strides; not inlined (see
// the macro kernel in blocked/gemm.h)
__attribute__((noinline)) void
direct(long m, long n, long k, double alpha, const double *A, long ldA,
       const double *B, long incRowB, long incColB, double beta, double *C,
       long ldC)
{
    const long m8 = m / 8 * 8, n4 = n / 4 * 4;
    for (long j = 0; j < n4; j += 4) {
	for (long i = 0; i < m8; i += 8) {
	    tile<4>(k, alpha, &A[i], ldA, &B[j * incColB], incRowB, incColB,
	            beta, &C[i + j * ldC], ldC);
	}
    }
    rest(m8, m, 0, n4, k, alpha, A, ldA, B, incRowB, incColB, beta, C, ldC);
    rest(0, m, n4, n, k, alpha, A, ldA, B, incRowB, incColB, beta, C, ldC);
}

int
main()
{
    struct
    {
	long m, n, k;
    } shapes[] = {{8, 8, 8},       {16, 16, 16},    {32, 32, 32},
                  {64, 64, 64},    {128, 128, 128}, {256, 256, 256},
                  {512, 512, 512}, {200, 4, 4},     {4, 200, 200},
                  {1000, 8, 8}};
    std::printf("%5s %5s %5s %9s %9s %9s   (GFLOPS)\n", "m", "n", "k", "direct",
                "blocked", "ulmBLAS");
    for (const auto &s : shapes) {
	const long m = s.m, n = s.n, k = s.k;
	std::vector<double> A(m * k), B(k * n), C(m * n), R(m * n);
	randomMatrix(m, k, A.data(), 1, m, 1);
	randomMatrix(k, n, B.data(), 1, k, 2);
	ulmBLAS::gemm(m, n, k, 1.0, false, A.data(), 1L, m, false, B.data(), 1L,
	              k, 0.0, R.data(), 1L, m);
	// repetitions: about 2e8 operations per measurement
	const long reps = std::max(1L, long(1e8 / (double(m) * n * k)));
	std::printf("%5ld %5ld %5ld", m, n, k);
	for (int v = 0; v < 3; ++v) {
	    double best = INFINITY;
	    for (int run = 0; run < 3; ++run) {
		lap();
		for (long r = 0; r < reps; ++r) {
		    if (v == 0) {
			direct(m, n, k, 1.0, A.data(), m, B.data(), 1, k, 0.0,
			       C.data(), m);
		    } else if (v == 1) {
			blocked::gemm<UK, 96, 256, 4104>(m, n, k, 1.0, A.data(),
			                                 1, m, B.data(), 1, k,
			                                 0.0, C.data(), 1, m);
		    } else {
			ulmBLAS::gemm(m, n, k, 1.0, false, A.data(), 1L, m,
			              false, B.data(), 1L, k, 0.0, C.data(), 1L,
			              m);
		    }
		}
		best = std::min(best, lap());
	    }
	    if (difference(m, n, C.data(), R.data(), m) > 1e-10) {
		std::printf(" %9s", "WRONG");
	    } else {
		std::printf(" %9.1f", 2e-9 * m * n * k * reps / best);
	    }
	}
	std::printf("\n");
    }
}
