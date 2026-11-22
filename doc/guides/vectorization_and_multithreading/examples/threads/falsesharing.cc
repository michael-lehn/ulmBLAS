// Every thread increments its own counter.  Adjacent counters share a
// cache line (false sharing); padded counters have one line each.
#include <atomic>
#include <cstdio>
#include <omp.h>

struct Adjacent
{
    std::atomic<long> value{0};
};

// aligned to 128 bytes and padded explicitly: the compiler has nothing
// to pad (MSVC (Microsoft Visual C++) would warn: C4324)
struct alignas(128) Padded
{
    std::atomic<long> value{0};
    char pad[128 - sizeof(std::atomic<long>)];
};
static_assert(sizeof(Padded) == 128, "one 128-byte block");

template <typename Counter>
double
run(const char *name)
{
    const int nt = 4;
    const long n = 20000000;
    Counter count[nt];

    const double t0 = omp_get_wtime();
#pragma omp parallel num_threads(nt)
    {
	std::atomic<long> &c = count[omp_get_thread_num()].value;
	for (long i = 0; i < n; ++i) {
	    c.fetch_add(1, std::memory_order_relaxed);
	}
    }
    const double t = omp_get_wtime() - t0;
    std::printf("%-8s sizeof = %3zu  %6.1f ms\n", name, sizeof(Counter),
                1e3 * t);
    return t;
}

int
main()
{
    run<Adjacent>("adjacent");
    run<Padded>("padded");
}
