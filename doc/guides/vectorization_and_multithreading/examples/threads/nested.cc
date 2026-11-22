// A library routine that uses threads only if it is not called from a
// parallel region (as gemm and the factorizations of ulmBLAS).
#include <cstdio>
#include <omp.h>

void
libraryRoutine(const char *caller)
{
    const int nt = omp_in_parallel() ? 1 : omp_get_max_threads();
#pragma omp parallel num_threads(nt)
    {
#pragma omp single
	std::printf("%-22s -> team of %d thread(s)\n", caller,
	            omp_get_num_threads());
    }
}

int
main()
{
    libraryRoutine("serial code");

#pragma omp parallel num_threads(2)
    {
	libraryRoutine(omp_get_thread_num() == 0 ? "parallel region, tid 0"
	                                         : "parallel region, tid 1");
    }
}
