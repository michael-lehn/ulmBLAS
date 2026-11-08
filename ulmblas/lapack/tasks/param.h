#ifndef ULMBLAS_LAPACK_TASKS_PARAM_H
#define ULMBLAS_LAPACK_TASKS_PARAM_H 1

//
//  Parameters of the task-based factorizations.  The defaults were tuned
//  for double on one socket of a Xeon Platinum 8358 (Ice Lake, 32 cores);
//  see experiments/lapack/README.md (Tuning).
//

#include <algorithm>

namespace ulmBLAS { namespace tasks {

struct Param
{
    long    nb;             // tile size
    long    ib;             // getrf: inner block of a panel
    int     panelThreads;   // getrf: max. threads of a panel (1: serial)
    int     threads;        // threads of the parallel region
    long    serialN;        // min(m,n) <= serialN: serial recursive algorithm
};

//  threads: the number of threads available (omp_get_max_threads())
inline Param
potrfParam(long n, int threads)
{
    Param p;
    p.nb           = (n<250) ? 32 : (n<600) ? 48 : (n<900) ? 64
                   : (n<2200) ? 96 : 128;
    p.ib           = 0;
    p.panelThreads = 1;
    p.serialN      = 100;
    //  about one thread per 64 rows (at least 4): more threads only add
    //  idle time and synchronization for small and medium n
    p.threads = std::max(1, std::min(threads, std::max(4, int(n/64))));
    //  more threads than tiles of the trailing matrix do not help
    const long nt    = (n+p.nb-1)/p.nb;
    const long tiles = nt*(nt+1)/2;
    p.threads = int(std::max(1L, std::min(long(p.threads), tiles)));
    return p;
}

inline Param
getrfParam(long m, long n, int threads)
{
    const long kmin = std::min(m, n);
    Param p;
    p.nb           = (kmin<600) ? 48 : (kmin<900) ? 64 : (kmin<2800) ? 96 : 128;
    p.ib           = 16;
    p.serialN      = 150;
    p.threads = std::max(1, std::min(threads, std::max(4, int(kmin/64))));
    const long mt  = (m+p.nb-1)/p.nb;
    const long nt  = (n+p.nb-1)/p.nb;
    p.threads      = int(std::max(1L, std::min(long(p.threads), mt*nt)));
    p.panelThreads = std::min(8, p.threads);
    return p;
}

} } // namespace tasks, ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_PARAM_H
