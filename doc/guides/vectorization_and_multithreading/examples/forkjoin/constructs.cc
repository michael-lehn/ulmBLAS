// Work sharing inside a parallel region: for with schedules, single,
// masked, barrier and nowait.
#include <cstdio>
#include <omp.h>

int
main()
{
    const int n = 16;
    int owner[n];

#pragma omp parallel num_threads(4)
    {
	const int tid = omp_get_thread_num();

	// static: contiguous chunks of n/4 iterations, fixed in advance
#pragma omp for schedule(static)
	for (int i = 0; i < n; ++i) {
	    owner[i] = tid;
	} // implicit barrier

#pragma omp single
	{
	    std::printf("static:     ");
	    for (int i = 0; i < n; ++i) {
		std::printf("%d", owner[i]);
	    }
	    std::printf("\n");
	} // implicit barrier

	// dynamic, 2: chunks of 2 iterations, handed out at run time
#pragma omp for schedule(dynamic, 2)
	for (int i = 0; i < n; ++i) {
	    owner[i] = tid;
	}

#pragma omp masked // only thread 0, no barrier
	{
	    std::printf("dynamic,2:  ");
	    for (int i = 0; i < n; ++i) {
		std::printf("%d", owner[i]);
	    }
	    std::printf("\n");
	}
#pragma omp barrier // the others must not overwrite owner too early

	// nowait: no barrier at the end of this loop
#pragma omp for schedule(static) nowait
	for (int i = 0; i < n; ++i) {
	    owner[i] = -1;
	}
    } // the implicit barrier at the end of the region remains
    std::printf("after:      owner[0] = %d\n", owner[0]);
}
