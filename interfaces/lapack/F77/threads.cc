#include LAPACK_HEADER
#include <ulmblas/lapack/threads.h>

//
//  Threads of the task-based LAPACK functions (?getrf_, ?potrf_), as in the
//  C interface.  The argument is passed by value, as for OpenBLAS's
//  openblas_set_num_threads, so that libblastrampoline (Julia) can call
//  them.  In Julia, before BLAS.lbt_forward:
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
    return ulmBLAS::numThreads();
}

} // extern "C"
