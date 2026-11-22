// A team of threads formed at run time from inside a task, as the panel
// factorization in ulmBLAS's LU (lower-upper) factorization
// (getrftasks.tcc): the task creates helper tasks, waits a short time for
// helpers to join, closes the team and works with the members it has.  It never
// waits for a thread that is busy elsewhere.
//
// The work: the maximum of |a(i,j)| of every column j of an m x n matrix
// (the pivot search of the LU factorization).  Every member searches its own
// rows; the members meet at a barrier per column.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <thread>
#include <vector>

const int CLOSED = 1 << 30;

class SpinBarrier // as in barrier.cc
{
  public:
    void
    reset(int n_)
    {
	n = n_;
    }

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
    int n = 1;
    alignas(64) std::atomic<int> count{0};
    alignas(64) std::atomic<int> sense{0};
};

struct Team
{
    long m, n;
    const double *A; // column major
    double *colMax;

    int creator; // the thread of the creating task
    int maxSize;
    std::atomic<int> state{1};   // members (0 is the creator), CLOSED bit
    std::atomic<int> started{0}; // the team is closed and set up
    int size = 1;
    SpinBarrier barrier;
    std::vector<double> cand; // [2][size]: candidates, double buffered
};

int
join(Team &T)
{
    int s = T.state.load(std::memory_order_acquire);
    while (!(s & CLOSED) && s < T.maxSize) {
	if (T.state.compare_exchange_weak(s, s + 1,
	                                  std::memory_order_acq_rel)) {
	    return s;
	}
    }
    return -1;
}

void
member(Team &T, int r)
{
    const long chunk = T.m / T.size + (T.m % T.size != 0);
    const long lo = std::min(T.m, r * chunk), hi = std::min(T.m, lo + chunk);
    int sense = 0;
    for (long j = 0; j < T.n; ++j) {
	const int par = int(j & 1);
	double v = 0;
	for (long i = lo; i < hi; ++i) {
	    v = std::fmax(v, std::fabs(T.A[i + j * T.m]));
	}
	T.cand[par * T.size + r] = v;
	T.barrier.wait(sense);
	// every member computes the maximum (here only member 0 stores it);
	// the next column uses the other half of cand, so no second barrier
	// is needed before cand[par] is written again
	double max = 0;
	for (int s = 0; s < T.size; ++s) {
	    max = std::fmax(max, T.cand[par * T.size + s]);
	}
	if (r == 0) {
	    T.colMax[j] = max;
	}
    }
}

// called inside a task; returns the size of the team
int
columnMaxima(long m, long n, const double *A, double *colMax, int maxSize,
             double joinWait)
{
    Team T;
    T.m = m;
    T.n = n;
    T.A = A;
    T.colMax = colMax;
    T.creator = omp_get_thread_num();
    T.maxSize = maxSize;

    Team *tp = &T;
    for (int h = 1; h < maxSize; ++h) {
#pragma omp task firstprivate(tp)
	{
	    if (omp_get_thread_num() != tp->creator) {
		const int r = join(*tp);
		if (r > 0) {
		    while (tp->started.load(std::memory_order_acquire) == 0) {
		    }
		    member(*tp, r);
		}
	    }
	}
    }
    // wait at most joinWait seconds for helpers, then close the team
    const double t0 = omp_get_wtime();
    while ((T.state.load(std::memory_order_acquire) & ~CLOSED) < maxSize &&
           omp_get_wtime() - t0 < joinWait) {
    }
    T.size = T.state.fetch_or(CLOSED, std::memory_order_acq_rel) & ~CLOSED;
    T.barrier.reset(T.size);
    T.cand.assign(2 * T.size, 0.0);
    T.started.store(1, std::memory_order_release);
    member(T, 0);
#pragma omp taskwait // the helper tasks must finish before T is destroyed
    return T.size;
}

int
main()
{
    const long m = 20000, n = 200;
    std::vector<double> A(m * n), colMax(n);
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < m; ++i) {
	    A[i + j * m] = std::sin(double(i * n + j));
	}
	A[(7 * j) % m + j * m] = -2.0 - double(j);
    }

    for (int busy = 0; busy <= 2; ++busy) {
	int size = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
	{
	    // busy threads: tasks that keep threads occupied for 50 ms
	    for (int b = 0; b < busy; ++b) {
#pragma omp task
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	    }
	    std::this_thread::sleep_for(std::chrono::milliseconds(5));
#pragma omp task shared(size)
	    size = columnMaxima(m, n, A.data(), colMax.data(), 4, 20e-6);
	}
	long errors = 0;
	for (long j = 0; j < n; ++j) {
	    errors += colMax[j] != 2.0 + double(j);
	}
	std::printf("%d busy thread(s): team of %d, errors: %ld\n", busy, size,
	            errors);
    }
}
