// compare_exchange_weak, twice as in the LU (lower-upper) factorization
// of ulmBLAS (getrftasks.tcc): an atomic minimum (the first zero pivot) and
// joining a team of bounded size that can be closed.
#include <atomic>
#include <climits>
#include <cstdio>
#include <omp.h>

// z = min(z, v)
void
atomicMin(std::atomic<long> &z, long v)
{
    long old = z.load();
    // on failure old is updated to the current value of z
    while (v < old && !z.compare_exchange_weak(old, v)) {
    }
}

const int CLOSED = 1 << 30;

// a member number 1, 2, ..., maxSize-1, or -1 (team full or closed)
int
join(std::atomic<int> &state, int maxSize)
{
    int s = state.load(std::memory_order_acquire);
    while (!(s & CLOSED) && s < maxSize) {
	if (state.compare_exchange_weak(s, s + 1, std::memory_order_acq_rel)) {
	    return s;
	}
    }
    return -1;
}

int
main()
{
    std::atomic<long> firstZero(LONG_MAX);
#pragma omp parallel num_threads(8)
    {
	// every thread reports some candidates
	const int t = omp_get_thread_num();
	for (long v = 1000 - 7 * t; v > 900; v -= 13) {
	    atomicMin(firstZero, v);
	}
    }
    std::printf("minimum: %ld\n", firstZero.load());

    // 8 threads try to join a team of at most 4 (member 0 exists already)
    std::atomic<int> state(1);
    std::atomic<int> joined(0), rejected(0);
#pragma omp parallel num_threads(8)
    {
	if (join(state, 4) > 0) {
	    ++joined;
	} else {
	    ++rejected;
	}
    }
    // close the team: no one can join afterwards
    const int size = state.fetch_or(CLOSED) & ~CLOSED;
    std::printf("joined: %d, rejected: %d, team size: %d, join after "
                "closing: %d\n",
                joined.load(), rejected.load(), size, join(state, 4));
}
