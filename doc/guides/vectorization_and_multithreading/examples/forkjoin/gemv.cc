// y = A x for a column-major m x n matrix A, split by rows: with
// "omp parallel for" and with an explicit partition in units of 8
// doubles (64 bytes, a cache line of y), as Level2Threads::range of
// ulmBLAS.
#include <cstdio>
#include <omp.h>
#include <vector>

// part tid of nt of [0, n) in whole units of u elements
void
range(int tid, int nt, long n, long u, long &i0, long &i1)
{
    const long units = n / u + (n % u != 0);
    i0 = units * tid / nt * u;       // always < n (for n > 0)
    i1 = units * (tid + 1) / nt * u; // the last part may end beyond n
    i1 = i1 < n ? i1 : n;
}

// rows i0, ..., i1-1 of y = A x: column by column (axpy)
void
gemvRows(long i0, long i1, long m, long n, const double *A, const double *x,
         double *y)
{
    for (long i = i0; i < i1; ++i) {
	y[i] = 0;
    }
    for (long j = 0; j < n; ++j) {
	const double xj = x[j];
	const double *a = &A[j * m];
#pragma omp simd
	for (long i = i0; i < i1; ++i) {
	    y[i] += a[i] * xj;
	}
    }
}

void
gemvParallelFor(long m, long n, const double *A, const double *x, double *y,
                int nt)
{
    // every thread gets a contiguous block of i, here as the loop over
    // the threads
#pragma omp parallel for num_threads(nt) schedule(static, 1)
    for (int t = 0; t < nt; ++t) {
	gemvRows(m * t / nt, m * (t + 1) / nt, m, n, A, x, y);
    }
}

void
gemvRegion(long m, long n, const double *A, const double *x, double *y, int nt)
{
#pragma omp parallel num_threads(nt)
    {
	long i0, i1;
	range(omp_get_thread_num(), omp_get_num_threads(), m, 8, i0, i1);
	gemvRows(i0, i1, m, n, A, x, y);
    }
}

int
main()
{
    const long m = 2000, n = 2000;
    std::vector<double> A(m * n), x(n), y(m), z(m);
    for (long j = 0; j < n; ++j) {
	x[j] = double(j % 4);
	for (long i = 0; i < m; ++i) {
	    A[j * m + i] = double((i + 2 * j) % 3);
	}
    }
    gemvRows(0, m, m, n, A.data(), x.data(), z.data()); // reference

    for (int nt = 1; nt <= 4; nt *= 2) {
	const int reps = 200;
	double t0 = omp_get_wtime();
	for (int r = 0; r < reps; ++r) {
	    gemvParallelFor(m, n, A.data(), x.data(), y.data(), nt);
	}
	double t1 = omp_get_wtime();
	const bool okFor = y == z;
	for (int r = 0; r < reps; ++r) {
	    gemvRegion(m, n, A.data(), x.data(), y.data(), nt);
	}
	double t2 = omp_get_wtime();
	const bool okRegion = y == z;
	std::printf("%d thread(s): parallel for %5.1f GFLOPS%s, region %5.1f "
	            "GFLOPS%s\n",
	            nt, 2e-9 * m * n * reps / (t1 - t0), okFor ? "" : " WRONG",
	            2e-9 * m * n * reps / (t2 - t1), okRegion ? "" : " WRONG");
    }
}
