// Producer/consumer with depend(out)/depend(in) on a token, where the
// consumer is created at about the time the producer finishes (varying
// delay in the generating thread).  If the producer has already finished,
// libomp takes the lock-free path (dn.task == NULL, no edge) and the
// consumer may start before the producer's stores are visible.
//
//   argv[1]: regions, argv[2]: threads, argv[3]: 1 = release fence at the
//   end of the producer (workaround), argv[4]: max delay (spin iterations)
//
// Apple M5, Homebrew libomp 23.1.4:
//   clang++ -std=c++17 -O2 test/libomp_fastpath.cc -Xpreprocessor -fopenmp \
//       -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
//   ./a.out 500000 4 0 2000    # 16M pairs: 21 and 12 errors in two runs
//   ./a.out 500000 4 1 2000    # with the fence: 0 errors
// Rare: about one error per million pairs.  The first test of this kind
// (one pair per region, consumer created right after the producer) never
// takes the lock-free path and shows nothing.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

constexpr int P = 32; // producer/consumer pairs per region
constexpr int N = 64; // ints written by a producer (4 cache lines)

alignas(64) static int a[P][N];
static std::atomic<int> done[P];
static char tok[P];
static std::atomic<long> sameThread{0};
static int gen;

static inline void
spin(int k)
{
    for (volatile int i = 0; i < k; ++i) {
    }
}

int
main(int argc, char **argv)
{
    long R = argc > 1 ? atol(argv[1]) : 20000;
    int T = argc > 2 ? atoi(argv[2]) : 4;
    int fence = argc > 3 ? atoi(argv[3]) : 0;
    int maxD = argc > 4 ? atoi(argv[4]) : 2000;

    std::atomic<long> errors{0}, errorsAfterDone{0};
    long createdAfterDone = 0, pairs = 0;

    for (long r = 1; r <= R; ++r) {
#pragma omp parallel num_threads(T)
#pragma omp single
	{
	    gen = omp_get_thread_num();
	    for (int p = 0; p < P; ++p) {
#pragma omp task depend(out : tok[p]) firstprivate(p, r)
		{
		    if (omp_get_thread_num() == gen)
			sameThread.fetch_add(1, std::memory_order_relaxed);
		    for (int i = 0; i < N; ++i)
			a[p][i] = int(r);
		    if (fence)
			std::atomic_thread_fence(std::memory_order_release);
		    done[p].store(int(r),
		                  std::memory_order_relaxed); // statistics only
		}
		spin(int((r * 7919 + p * 104729) % maxD));
		bool fin = done[p].load(std::memory_order_relaxed) == int(r);
		createdAfterDone += fin;
		++pairs;
#pragma omp task depend(in : tok[p]) firstprivate(p, r, fin)
		{
		    for (int i = 0; i < N; ++i) {
			if (a[p][i] != int(r)) {
			    errors.fetch_add(1, std::memory_order_relaxed);
			    if (fin)
				errorsAfterDone.fetch_add(
				    1, std::memory_order_relaxed);
			    break;
			}
		    }
		}
	    }
	}
    }
    std::printf("threads %d fence %d: %ld pairs, %ld consumers created after "
                "the producer finished, "
                "%ld errors (%ld of them in that case); producers on the "
                "generating thread: %ld\n",
                T, fence, pairs, createdAfterDone, errors.load(),
                errorsAfterDone.load(), sameThread.load());
    return errors.load() != 0;
}
