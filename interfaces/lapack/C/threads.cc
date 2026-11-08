#include LAPACK_HEADER
#include <ulmblas/lapack/threads.h>

//
//  Threads of the task-based LAPACK functions (?getrf, ?potrf): n > 0 sets
//  the number, n <= 0 restores the default (omp_get_max_threads()).  The
//  setting is global.  Without OpenMP the functions run serially and
//  get_num_threads returns 1.
//

extern "C" {

void
ULMLAPACK(set_num_threads)(int n)
{
    ulmBLAS::setNumThreads(n);
}

int
ULMLAPACK(get_num_threads)(void)
{
    return ulmBLAS::numThreads();
}

} // extern "C"
