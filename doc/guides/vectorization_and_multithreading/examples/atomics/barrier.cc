// Two spin barriers built from atomics (as ulmBLAS's Barrier in
// auxiliary/threadpool.h and SpinBarrier in lapack/tasks/sync.h),
// tested and compared with the barrier of OpenMP.
#include <atomic>
#include <cstdio>
#include <omp.h>
#include <vector>

// a counter and a generation number
class Barrier
{
  public:
    explicit Barrier(int n) : n(n) {}

    void
    wait()
    {
	const int gen = generation.load(std::memory_order_acquire);
	if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
	    count.store(0, std::memory_order_relaxed); // the last one
	    generation.fetch_add(1, std::memory_order_release);
	    return;
	}
	while (generation.load(std::memory_order_acquire) == gen) {
	}
    }

  private:
    const int n;
    alignas(64) std::atomic<int> count{0};
    alignas(64) std::atomic<int> generation{0};
};

// sense reversing: every thread flips its own sense in each round
class SpinBarrier
{
  public:
    explicit SpinBarrier(int n) : n(n) {}

    void
    wait(int &localSense)
    {
	localSense = 1 - localSense;
	if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
	    count.store(0, std::memory_order_relaxed);
	    sense.store(localSense, std::memory_order_release);
	} else {
	    while (sense.load(std::memory_order_acquire) != localSense) {
	    }
	}
    }

  private:
    const int n;
    alignas(64) std::atomic<int> count{0};
    alignas(64) std::atomic<int> sense{0};
};

// rounds: thread t writes round r into its slot; after the barrier every
// thread checks all slots; after a second barrier the next round starts
template <typename Wait>
void
test(const char *name, int nt, long rounds, Wait wait)
{
    std::vector<long> slot(nt * 16, -1); // 16 longs: 128 bytes apart
    long errors = 0;
    const double t0 = omp_get_wtime();
#pragma omp parallel num_threads(nt) reduction(+ : errors)
    {
	const int t = omp_get_thread_num();
	int sense = 0;
	for (long r = 0; r < rounds; ++r) {
	    slot[t * 16] = r;
	    wait(sense);
	    for (int u = 0; u < nt; ++u) {
		errors += slot[u * 16] != r;
	    }
	    wait(sense);
	}
    }
    const double t = omp_get_wtime() - t0;
    std::printf("%-12s errors: %ld, %.0f ns per barrier\n", name, errors,
                1e9 * t / (2 * rounds));
}

int
main()
{
    const int nt = 4;
    const long rounds = 1000000;
    Barrier barrier(nt);
    SpinBarrier spinBarrier(nt);

    test("Barrier", nt, rounds, [&](int &) { barrier.wait(); });
    test("SpinBarrier", nt, rounds, [&](int &s) { spinBarrier.wait(s); });
    test("omp barrier", nt, rounds, [](int &) {
#pragma omp barrier
    });
}
