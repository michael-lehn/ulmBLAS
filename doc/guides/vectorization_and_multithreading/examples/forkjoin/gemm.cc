// C += A B on the thread pool of pool.h with the protocol of ulmBLAS's
// parallel gemm (GemmFrame::runLoops in ulmblas/level3/gemm/frame.h),
// simplified to one group of threads:
//
//   - the k dimension is processed in blocks s = 0, 1, ... of kc rows of B;
//   - block s of B is packed into buffer s mod 2 (double buffering), by
//     all threads together: thread u packs share u (a range of columns);
//   - thread u computes its rows of C with all shares of the block, its
//     own share right away, the others as soon as they are ready.
//
// Two counters per thread, each in a cache line of its own:
//
//   ready[u] = s+1   share u of block s is packed
//   done[u]  = s+1   thread u has finished block s
//
// A thread packs its share of block s only when every thread has finished
// block s-2, the last user of the same buffer (done[v] >= s-1).
#include "pool.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

struct alignas(128) Flag
{
    std::atomic<long> value{0};
    char pad[128 - sizeof(std::atomic<long>)];

    void
    waitFor(long v) const
    {
	while (value.load(std::memory_order_acquire) < v) {
	}
    }
};
static_assert(sizeof(Flag) == 128, "one 128-byte block");

// part u of nt of [0, n) in whole units of r
void
range(int u, int nt, long n, long r, long &i0, long &i1)
{
    const long units = n / r + (n % r != 0);
    i0 = units * u / nt * r; // always < n (for n > 0)
    i1 = std::min(n, units * (u + 1) / nt * r);
}

// C(i0:i1, j0:j1) += A(i0:i1, l:l+kc) Bs(:, j0:j1); row major, Bs is kc x n
void
kernel(long i0, long i1, long j0, long j1, long l, long kc, long n, long k,
       const double *A, const double *Bs, double *C)
{
    for (long i = i0; i < i1; ++i) {
	for (long p = 0; p < kc; ++p) {
	    const double a = A[i * k + l + p];
#pragma omp simd
	    for (long j = j0; j < j1; ++j) {
		C[i * n + j] += a * Bs[p * n + j];
	    }
	}
    }
}

void
gemm(ThreadPool &pool, int nt, long m, long n, long k, const double *A,
     const double *B, double *C)
{
    const long KC = 256;
    std::vector<double> buffer(2 * KC * n);
    std::vector<Flag> ready(nt), done(nt);

    auto body = [&](int u, int threads) {
	if (threads != nt) { // fewer threads than planned: all on thread 0
	    for (long l = 0; l < k; l += KC) {
		const long kc = std::min(KC, k - l);
		std::copy(&B[l * n], &B[(l + kc) * n], buffer.data());
		kernel(0, m, 0, n, l, kc, n, k, A, buffer.data(), C);
	    }
	    return;
	}
	long i0, i1, c0, c1;
	range(u, nt, m, 4, i0, i1);
	long s = 0;
	for (long l = 0; l < k; l += KC, ++s) {
	    const long kc = std::min(KC, k - l);
	    double *Bs = &buffer[(s % 2) * KC * n];

	    // the buffer is free when block s-2 is finished everywhere
	    if (s >= 2) {
		for (int v = 0; v < nt; ++v) {
		    done[v].waitFor(s - 1);
		}
	    }
	    // pack the own share: columns c0, ..., c1-1 of rows l, ..., l+kc-1
	    range(u, nt, n, 8, c0, c1);
	    for (long p = 0; p < kc; ++p) {
		std::copy(&B[(l + p) * n + c0], &B[(l + p) * n + c1],
		          &Bs[p * n + c0]);
	    }
	    ready[u].value.store(s + 1, std::memory_order_release);

	    // the own share right away (packed by this thread: no wait), then
	    // the others as soon as they are ready
	    kernel(i0, i1, c0, c1, l, kc, n, k, A, Bs, C);
	    for (int t = 1; t < nt; ++t) {
		const int c = (u + t) % nt;
		ready[c].waitFor(s + 1);
		range(c, nt, n, 8, c0, c1);
		kernel(i0, i1, c0, c1, l, kc, n, k, A, Bs, C);
	    }
	    done[u].value.store(s + 1, std::memory_order_release);
	}
    };
    pool.run(nt, body);
}

int
main()
{
    const long m = 1000, n = 1000, k = 1000;
    std::vector<double> A(m * k), B(k * n), C(m * n), R(m * n);
    for (long i = 0; i < m * k; ++i) {
	A[i] = double(i % 7) - 3;
    }
    for (long i = 0; i < k * n; ++i) {
	B[i] = double(i % 5) - 2;
    }

    ThreadPool pool(4);
    std::fill(R.begin(), R.end(), 0.0);
    gemm(pool, 1, m, n, k, A.data(), B.data(), R.data()); // reference

    for (int nt = 1; nt <= 4; nt *= 2) {
	std::fill(C.begin(), C.end(), 0.0);
	const auto t0 = std::chrono::steady_clock::now();
	gemm(pool, nt, m, n, k, A.data(), B.data(), C.data());
	const auto t1 = std::chrono::steady_clock::now();
	const double t = std::chrono::duration<double>(t1 - t0).count();
	std::printf("%d thread(s): %5.1f GFLOPS, %s\n", nt,
	            2e-9 * m * n * k / t, C == R ? "correct" : "WRONG");
    }
}
