// A counter incremented by four threads: a data race and three fixes.
#include <atomic>
#include <cstdio>
#include <omp.h>

int
main()
{
    const long n = 10000000;

    long racy = 0;
    double t0 = omp_get_wtime();
#pragma omp parallel for num_threads(4)
    for (long i = 0; i < n; ++i) {
	++racy; // data race: undefined behavior
    }
    double t1 = omp_get_wtime();
    std::printf("racy:      %9ld  %6.1f ms\n", racy, 1e3 * (t1 - t0));

    // volatile: every increment is a load and a store (still a data race)
    volatile long racyVolatile = 0;
    t0 = omp_get_wtime();
#pragma omp parallel for num_threads(4)
    for (long i = 0; i < n; ++i) {
	racyVolatile = racyVolatile + 1;
    }
    t1 = omp_get_wtime();
    std::printf("volatile:  %9ld  %6.1f ms\n", racyVolatile, 1e3 * (t1 - t0));

    std::atomic<long> counter(0);
    t0 = omp_get_wtime();
#pragma omp parallel for num_threads(4)
    for (long i = 0; i < n; ++i) {
	counter.fetch_add(1, std::memory_order_relaxed);
    }
    t1 = omp_get_wtime();
    std::printf("atomic:    %9ld  %6.1f ms\n", counter.load(), 1e3 * (t1 - t0));

    long ompAtomic = 0;
    t0 = omp_get_wtime();
#pragma omp parallel for num_threads(4)
    for (long i = 0; i < n; ++i) {
#pragma omp atomic
	++ompAtomic;
    }
    t1 = omp_get_wtime();
    std::printf("omp atomic:%9ld  %6.1f ms\n", ompAtomic, 1e3 * (t1 - t0));

    long sum = 0;
    t0 = omp_get_wtime();
#pragma omp parallel for num_threads(4) reduction(+ : sum)
    for (long i = 0; i < n; ++i) {
	++sum; // every thread increments its own copy
    }
    t1 = omp_get_wtime();
    std::printf("reduction: %9ld  %6.1f ms\n", sum, 1e3 * (t1 - t0));
}
