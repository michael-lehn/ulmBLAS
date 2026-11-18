#include BLAS_HEADER
#include <cstddef>
#include <ulmblas/auxiliary/threadpool.h>
#include <ulmblas/lapack/threads.h>

//
//  Threads of ulmBLAS, as in the C interface (cblas.h).  The arguments are
//  passed by value, as for OpenBLAS's openblas_set_num_threads, so that
//  libblastrampoline (Julia) can call them.  In Julia, before
//  BLAS.lbt_forward:
//
//      lbt_register_thread_interface("ulmblas_get_num_threads_",
//                                    "ulmblas_set_num_threads_")
//
//  (libblastrampoline appends the suffix of the library, 64_); then
//  BLAS.get_num_threads and BLAS.set_num_threads include ulmBLAS.
//

extern "C" {

void
F77BLAS(ulmblas_set_num_threads)(int n)
{
    ulmBLAS::setNumThreads(n);
}

int
F77BLAS(ulmblas_get_num_threads)(void)
{
    return int(ulmBLAS::ThreadPool::instance().size());
}

void
F77BLAS(ulmblas_set_affinity)(std::ptrdiff_t mode)
{
    ulmBLAS::threads::Affinity::instance().set(mode);
}

std::ptrdiff_t
F77BLAS(ulmblas_get_affinity)(void)
{
    return ulmBLAS::threads::Affinity::instance().mode.load(
        std::memory_order_relaxed);
}

} // extern "C"
