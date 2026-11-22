// Fork-join with atomic flags, the pattern of the parallel gemm of
// ulmBLAS in miniature: y = A*(alpha*x).
//
// Every thread prepares one share of the buffer b = alpha*x (in gemm:
// packs one share of a block of B) and announces it with a flag.  Then it
// multiplies its rows of A with the whole buffer: its own share at once,
// every other share as soon as its flag is set.  No thread waits for all
// the others, as it would at a barrier.
#include <atomic>
#include <cstdio>
#include <omp.h>
#include <vector>

struct alignas(128) Flag // a cache line of its own
{
    std::atomic<int> value{0};
};

// part u of nt of [0, n)
void
range(int u, int nt, long n, long &i0, long &i1)
{
    i0 = n * u / nt;
    i1 = n * (u + 1) / nt;
}

void
scaledProduct(long m, long n, double alpha, const double *A, const double *x,
              double *b, double *y, int nt)
{
    std::vector<Flag> ready(nt);

#pragma omp parallel num_threads(nt)
    {
	const int u = omp_get_thread_num();

	// 1. prepare the own share of b and announce it
	long j0, j1;
	range(u, nt, n, j0, j1);
	for (long j = j0; j < j1; ++j) {
	    b[j] = alpha * x[j];
	}
	ready[u].value.store(1, std::memory_order_release);

	// 2. the own rows of y with the own share: written by this thread,
	//    so no flag is needed, and y[i] is set rather than updated
	long i0, i1;
	range(u, nt, m, i0, i1);
	for (long i = i0; i < i1; ++i) {
	    double s = 0;
	    for (long j = j0; j < j1; ++j) {
		s += A[i * n + j] * b[j];
	    }
	    y[i] = s;
	}

	// 3. the other shares u+1, ..., u-1, each as soon as it is ready
	for (int t = 1; t < nt; ++t) {
	    const int c = (u + t) % nt;
	    while (ready[c].value.load(std::memory_order_acquire) == 0) {
		// wait until share c is ready
	    }
	    range(c, nt, n, j0, j1);
	    for (long i = i0; i < i1; ++i) {
		double s = 0;
		for (long j = j0; j < j1; ++j) {
		    s += A[i * n + j] * b[j];
		}
		y[i] += s;
	    }
	}
    }
}

int
main()
{
    const long m = 1000, n = 800;
    std::vector<double> A(m * n), x(n), b(n), y(m);
    for (long i = 0; i < m; ++i) {
	for (long j = 0; j < n; ++j) {
	    A[i * n + j] = double((i + j) % 5); // small integers: exact sums
	}
    }
    for (long j = 0; j < n; ++j) {
	x[j] = double(j % 3);
    }

    scaledProduct(m, n, 2.0, A.data(), x.data(), b.data(), y.data(), 4);

    long errors = 0;
    for (long i = 0; i < m; ++i) {
	double s = 0;
	for (long j = 0; j < n; ++j) {
	    s += A[i * n + j] * 2.0 * x[j];
	}
	errors += s != y[i];
    }
    std::printf("m = %ld, n = %ld, wrong elements of y: %ld\n", m, n, errors);
}
