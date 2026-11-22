// The thread pool of pool.h: a test and the cost of an empty parallel
// region, compared with "omp parallel".
#include "pool.h"
#include <chrono>
#include <cstdio>
#include <omp.h>

double
seconds()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int
main()
{
    const int nt = 4;
    ThreadPool pool(nt);

    // every thread marks its slot; a nested call runs on one thread
    int mark[nt] = {}, nested[nt] = {};
    auto body = [&](int tid, int threads) {
	mark[tid] = threads;
	auto inner = [&](int, int innerThreads) { nested[tid] = innerThreads; };
	pool.run(nt, inner);
    };
    pool.run(nt, body);
    for (int t = 0; t < nt; ++t) {
	std::printf("thread %d: team of %d, nested team of %d\n", t, mark[t],
	            nested[t]);
    }

    const int reps = 100000;
    // asm volatile: the compiler must not remove the empty regions
    auto empty = [](int, int) { asm volatile("" : : : "memory"); };
    double t0 = seconds();
    for (int r = 0; r < reps; ++r) {
	pool.run(nt, empty);
    }
    double t1 = seconds();
    for (int r = 0; r < reps; ++r) {
#pragma omp parallel num_threads(nt)
	{
	    asm volatile("" : : : "memory");
	}
    }
    double t2 = seconds();
    std::printf("empty region: pool %.2f us, omp parallel %.2f us\n",
                1e6 * (t1 - t0) / reps, 1e6 * (t2 - t1) / reps);
}
