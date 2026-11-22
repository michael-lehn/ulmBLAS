// A parallel region: a team of threads runs the same block.
#include <cstdio>
#include <omp.h>

int
main()
{
    int shared = 42; // declared outside: shared by the team
    int x = 1;

#pragma omp parallel num_threads(4) firstprivate(x)
    {
	// declared inside: private, every thread has its own
	const int tid = omp_get_thread_num();
	const int nt = omp_get_num_threads();
	x += tid; // firstprivate: a private copy, initialized with 1

#pragma omp critical
	std::printf("thread %d of %d: shared = %d, x = %d\n", tid, nt, shared,
	            x);
    } // implicit barrier: the team joins

    std::printf("after the region: %d thread, x = %d\n", omp_get_num_threads(),
                x);
}
