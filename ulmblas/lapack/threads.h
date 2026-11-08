#ifndef ULMBLAS_LAPACK_THREADS_H
#define ULMBLAS_LAPACK_THREADS_H 1

//
//  Number of threads of the task-based LAPACK functions (getrf, potrf).
//
//      setNumThreads(n)   n > 0: use n threads; n <= 0: the default
//      numThreads()       the number used: the value set, by default
//                         omp_get_max_threads(); 1 without OpenMP
//
//  The setting is global (all threads of the program), unlike
//  omp_set_num_threads, which only affects the calling thread.  The C and
//  Fortran interfaces export it (ulmblas_set_num_threads, ...), so that
//  e.g. libblastrampoline (Julia) can set it together with the threads of
//  other BLAS libraries.
//

#include <atomic>

#ifdef _OPENMP
#   include <omp.h>
#endif

namespace ulmBLAS {

inline std::atomic<int> &
numThreadsSetting()
{
    static std::atomic<int> n(0);
    return n;
}

inline void
setNumThreads(int n)
{
    numThreadsSetting().store(n>0 ? n : 0, std::memory_order_relaxed);
}

inline int
numThreads()
{
#ifdef _OPENMP
    const int n = numThreadsSetting().load(std::memory_order_relaxed);
    return n>0 ? n : omp_get_max_threads();
#else
    return 1;
#endif
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_THREADS_H
