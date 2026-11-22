// Tasks with dependencies and an atomic flag, the pattern of the
// Cholesky factorization of ulmBLAS (potrfTasks) in miniature: here every
// tile is a single element.
//
// A = L L^T, A symmetric positive definite, n x n, row major; the lower
// triangle is overwritten with L.  Step k of the right-looking algorithm:
//
// (POTRF: positive definite triangular factorization, TRSM: triangular
// solve with multiple right-hand sides, SYRK: symmetric rank-k update,
// GEMM: general matrix-matrix product)
//
//   POTRF  a(k,k) = sqrt(a(k,k))                    inout a(k,k)
//   TRSM   a(i,k) = a(i,k) / a(k,k),       i > k    in a(k,k),  inout a(i,k)
//   SYRK   a(i,i) -= a(i,k)^2,             i > k    in a(i,k),  inout a(i,i)
//   GEMM   a(i,j) -= a(i,k) a(j,k),    k < j < i    in a(i,k), a(j,k),
//                                                   inout a(i,j)
//
// If a(k,k) <= 0, A is not positive definite: the step k is stored in an
// atomic variable and the tasks of the steps k, k+1, ... do nothing.
// Returns 0 or k+1 (as the INFO (information) argument of LAPACK
// (Linear Algebra Package)).
#include <atomic>
#include <cmath>
#include <cstdio>
#include <vector>

long
cholesky(long n, double *A)
{
    std::atomic<long> failStep(n); // n: no failure
    long info = 0;
    auto run = [&](long k) {
	return failStep.load(std::memory_order_relaxed) > k;
    };
    auto a = [=](long i, long j) -> double & { return A[i * n + j]; };

#pragma omp parallel
#pragma omp single
    for (long k = 0; k < n; ++k) {
#pragma omp task depend(inout : a(k, k))
	{
	    if (run(k)) {
		if (a(k, k) <= 0) {
		    info = k + 1;
		    failStep.store(k, std::memory_order_relaxed);
		} else {
		    a(k, k) = std::sqrt(a(k, k));
		}
	    }
	}
	for (long i = k + 1; i < n; ++i) {
#pragma omp task depend(in : a(k, k)) depend(inout : a(i, k))
	    if (run(k)) {
		a(i, k) /= a(k, k);
	    }
	}
	for (long i = k + 1; i < n; ++i) {
#pragma omp task depend(in : a(i, k)) depend(inout : a(i, i))
	    if (run(k)) {
		a(i, i) -= a(i, k) * a(i, k);
	    }
	    for (long j = k + 1; j < i; ++j) {
#pragma omp task depend(in : a(i, k), a(j, k)) depend(inout : a(i, j))
		if (run(k)) {
		    a(i, j) -= a(i, k) * a(j, k);
		}
	    }
	}
    }
    return info;
}

int
main()
{
    const long n = 60;
    std::vector<double> A(n * n), L(n * n);
    for (long i = 0; i < n; ++i) {
	for (long j = 0; j < n; ++j) {
	    A[i * n + j] = (i == j) ? n + 1.0 : 1.0 / (1 + i + j);
	}
    }
    L = A;
    const long info = cholesky(n, L.data());

    // the residual max |A - L L^T| (lower triangle)
    double err = 0;
    for (long i = 0; i < n; ++i) {
	for (long j = 0; j <= i; ++j) {
	    double s = 0;
	    for (long l = 0; l <= j; ++l) {
		s += L[i * n + l] * L[j * n + l];
	    }
	    err = std::fmax(err, std::fabs(A[i * n + j] - s));
	}
    }
    std::printf("positive definite: info = %ld, max |A - L L^T| = %.2e\n", info,
                err);

    // not positive definite: a(10,10) is too small
    L = A;
    L[10 * n + 10] = -1;
    std::printf("not positive definite: info = %ld\n", cholesky(n, L.data()));
}
