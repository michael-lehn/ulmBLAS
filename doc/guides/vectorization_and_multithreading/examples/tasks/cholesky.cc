// Tiled Cholesky factorization with tasks, dependencies and priorities,
// structured as potrfTasks of ulmBLAS (ulmblas/lapack/tasks/potrftasks.tcc).
// A is n x n, column major with leading dimension n, symmetric positive
// definite; its lower triangle is overwritten with L, A = L L^T.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <omp.h>
#include <vector>

// ---- tile kernels --------------------------------------------------
// simple loops; ulmBLAS calls its BLAS (Basic Linear Algebra
// Subprograms) here

// A = L L^T for an nk x nk tile; 0 or the column (1-based) that fails
long
potf2(long nk, double *A, long ld)
{
    for (long j = 0; j < nk; ++j) {
	double d = A[j + j * ld];
	for (long p = 0; p < j; ++p) {
	    d -= A[j + p * ld] * A[j + p * ld];
	}
	if (!(d > 0)) {
	    return j + 1;
	}
	d = std::sqrt(d);
	A[j + j * ld] = d;
	for (long p = 0; p < j; ++p) {
	    const double ljp = A[j + p * ld];
#pragma omp simd
	    for (long i = j + 1; i < nk; ++i) {
		A[i + j * ld] -= A[i + p * ld] * ljp;
	    }
	}
#pragma omp simd
	for (long i = j + 1; i < nk; ++i) {
	    A[i + j * ld] /= d;
	}
    }
    return 0;
}

// B = B L^{-T}, L lower triangular nk x nk, B ni x nk
void
trsm(long ni, long nk, const double *L, double *B, long ld)
{
    for (long j = 0; j < nk; ++j) {
	for (long p = 0; p < j; ++p) {
	    const double ljp = L[j + p * ld];
#pragma omp simd
	    for (long i = 0; i < ni; ++i) {
		B[i + j * ld] -= B[i + p * ld] * ljp;
	    }
	}
	const double d = L[j + j * ld];
#pragma omp simd
	for (long i = 0; i < ni; ++i) {
	    B[i + j * ld] /= d;
	}
    }
}

// C = C - A B^T, C mi x nj, A mi x nk, B nj x nk; lowerOnly: only the
// lower triangle of C (the update of a diagonal tile)
void
gemmNT(long mi, long nj, long nk, const double *A, const double *B, double *C,
       long ld, bool lowerOnly)
{
    for (long j = 0; j < nj; ++j) {
	const long i0 = lowerOnly ? j : 0;
	for (long p = 0; p < nk; ++p) {
	    const double b = B[j + p * ld];
#pragma omp simd
	    for (long i = i0; i < mi; ++i) {
		C[i + j * ld] -= A[i + p * ld] * b;
	    }
	}
    }
}

// ---- the task graph ---------------------------------------------------

long
potrfTasks(long n, double *A, long nb, int threads)
{
    const int PRIO_POTRF = 3, PRIO_CRIT = 2, PRIO_NEXT = 1, PRIO_OTHER = 0;
    const long nt = n / nb + (n % nb != 0);
    const long ld = n;
    auto tile = [=](long i, long j) { return &A[i * nb + j * nb * ld]; };

    std::atomic<long> failStep(std::numeric_limits<long>::max());
    long info = 0;
    auto run = [&](long k) {
	return failStep.load(std::memory_order_relaxed) > k;
    };

    // tile (i,j) in the depend clauses: its first element
#define TILE(i, j) A[(i) * nb + (j) * nb * ld]

#pragma omp parallel num_threads(threads)
#pragma omp single
    for (long k = 0; k < nt; ++k) {
	const long nk = std::min(nb, n - k * nb);
	double *Akk = tile(k, k);

#pragma omp task depend(inout : TILE(k, k)) priority(PRIO_POTRF)
	if (run(k)) {
	    const long inf = potf2(nk, Akk, ld);
	    if (inf) {
		info = k * nb + inf;
		failStep.store(k, std::memory_order_relaxed);
	    }
	}

	for (long i = k + 1; i < nt; ++i) {
	    const long ni = std::min(nb, n - i * nb);
	    double *Aik = tile(i, k);
#pragma omp task depend(in : TILE(k, k)) depend(inout : TILE(i, k))            \
    priority(i == k + 1 ? PRIO_POTRF : PRIO_CRIT)
	    if (run(k)) {
		trsm(ni, nk, Akk, Aik, ld);
	    }
	}

	for (long i = k + 1; i < nt; ++i) {
	    const long ni = std::min(nb, n - i * nb);
	    const double *Aik = tile(i, k);
	    double *Aii = tile(i, i);
#pragma omp task depend(in : TILE(i, k)) depend(inout : TILE(i, i))            \
    priority(i == k + 1 ? PRIO_POTRF : PRIO_NEXT)
	    if (run(k)) {
		gemmNT(ni, ni, nk, Aik, Aik, Aii, ld, true);
	    }

	    for (long j = k + 1; j < i; ++j) {
		const long nj = std::min(nb, n - j * nb);
		const double *Ajk = tile(j, k);
		double *Aij = tile(i, j);
#pragma omp task depend(in : TILE(i, k), TILE(j, k))                           \
    depend(inout : TILE(i, j)) priority(j == k + 1 ? PRIO_CRIT : PRIO_OTHER)
		if (run(k)) {
		    gemmNT(ni, nj, nk, Aik, Ajk, Aij, ld, false);
		}
	    }
	}
    }
#undef TILE
    return info;
}

// max |A - L L^T| / max |A| over the lower triangle
double
residual(long n, const double *A, const double *L)
{
    double err = 0, norm = 0;
    for (long j = 0; j < n; ++j) {
	for (long i = j; i < n; ++i) {
	    double s = 0;
	    for (long p = 0; p <= j; ++p) {
		s += L[i + p * n] * L[j + p * n];
	    }
	    err = std::max(err, std::fabs(A[i + j * n] - s));
	    norm = std::max(norm, std::fabs(A[i + j * n]));
	}
    }
    return err / norm;
}

int
main()
{
    const long n = 2000, nb = 128;
    std::vector<double> A(n * n), L(n * n);
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < n; ++i) {
	    A[i + j * n] = (i == j) ? double(n) : 1.0 / (1.0 + i + j);
	}
    }

    for (int threads = 1; threads <= 4; threads *= 2) {
	L = A;
	const double t0 = omp_get_wtime();
	const long info = potrfTasks(n, L.data(), nb, threads);
	const double t = omp_get_wtime() - t0;
	std::printf("%d thread(s): %6.1f ms, %5.1f GFLOPS, info = %ld\n",
	            threads, 1e3 * t, 1e-9 * n * n * n / 3 / t, info);
    }
    std::printf("relative residual: %.1e\n", residual(n, A.data(), L.data()));

    L = A;
    L[1500 + 1500 * n] = -1; // not positive definite
    std::printf("not positive definite: info = %ld\n",
                potrfTasks(n, L.data(), nb, 4));
}
