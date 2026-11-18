#include BLAS_HEADER
#include <cstddef>
#include <ulmblas/auxiliary/threadpool.h>
#include <ulmblas/lapack/threads.h>

//
//  Threads of ulmBLAS (cblas.h): the number of threads of the BLAS
//  functions (thread pool) and of the task-based LAPACK functions (?getrf,
//  ?potrf with OpenMP); n > 0 sets the number, n <= 0 restores the default.
//  Affinity (Linux only): 1 pins the threads of a parallel region to one
//  CPU each (one per core first), 2 (default) pins them for one call after
//  a change and then unpins them (placed once), 0 leaves them to the
//  operating system.  The settings are global and take effect at the next call.
//

extern "C" {

void
ULMBLAS(set_num_threads)(int n)
{
    ulmBLAS::setNumThreads(n);
}

int
ULMBLAS(get_num_threads)(void)
{
    return int(ulmBLAS::ThreadPool::instance().size());
}

void
ULMBLAS(set_affinity)(std::ptrdiff_t mode)
{
    ulmBLAS::threads::Affinity::instance().set(mode);
}

std::ptrdiff_t
ULMBLAS(get_affinity)(void)
{
    return ulmBLAS::threads::Affinity::instance().mode.load(
        std::memory_order_relaxed);
}

} // extern "C"
