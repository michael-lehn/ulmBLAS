// Tasks: created by one thread (single), run by any thread of the team.
#include <cstdio>
#include <omp.h>
#include <vector>

// recursive sum: two tasks per call, taskwait for both
long
sum(const long *x, long n)
{
    if (n <= 1000) { // cutoff: small sums without tasks
	long s = 0;
	for (long i = 0; i < n; ++i) {
	    s += x[i];
	}
	return s;
    }
    long a, b;
#pragma omp task shared(a)
    a = sum(x, n / 2);
#pragma omp task shared(b)
    b = sum(x + n / 2, n - n / 2);
#pragma omp taskwait // a and b are computed now
    return a + b;
}

int
main()
{
#pragma omp parallel num_threads(4)
#pragma omp single
    {
	// i is firstprivate in each task: its value at creation
	for (int i = 0; i < 4; ++i) {
#pragma omp task
	    std::printf("task %d runs on thread %d\n", i, omp_get_thread_num());
	}
    } // implicit barrier: all tasks are finished here

    const long n = 1000000;
    std::vector<long> x(n);
    for (long i = 0; i < n; ++i) {
	x[i] = i;
    }
    long s = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
    s = sum(x.data(), n);
    std::printf("sum = %ld (expected %ld)\n", s, n * (n - 1) / 2);
}
