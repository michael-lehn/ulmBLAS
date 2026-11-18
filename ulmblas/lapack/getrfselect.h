#ifndef ULMBLAS_LAPACK_GETRFSELECT_H
#define ULMBLAS_LAPACK_GETRFSELECT_H 1

#include <algorithm>
#include <complex>
#include <cstddef>
#include <ulmblas/auxiliary/threadpool.h>

#ifdef ULMBLAS_GETRF_STUDY
#include <cstdlib>
#endif

namespace ulmBLAS {

//
//  The algorithm of getrf (ulmblas/lapack/getrf.tcc) for an m x n matrix,
//  k = min(m, n):
//
//      Recursive   getrf2 (recursive halving); the level 3 calls use the
//                  threads of the pool
//      Lookahead   getrfLookahead (getrfla.h): block size nb, lookahead
//                  of one panel, on threads threads
//      Tasks       OPENMP=1 only: the task graph (tasks/getrftasks.h),
//                  from TasksMinThreads threads on
//
//  Initial values (Intel i5-12500 6 cores, Xeon 8358 32 and 64 threads, d
//  and z, see investigations/2026-11-getrf) until make tune measures them:
//  one thread per MinCols columns of k, at most the pool; lookahead from
//  LookaheadMinK to LookaheadMaxCols columns per thread, adaptive panel
//  widths between NBMin and NB.
//
struct GetrfPlan
{
    enum Variant
    {
	Recursive,
	Lookahead,
	Tasks
    };

    Variant variant;
    std::ptrdiff_t nb, nbMin, chunk;
    std::ptrdiff_t threads;
};

template <typename T> struct GetrfParam
{
    static constexpr std::ptrdiff_t MinCols = 64;
    static constexpr std::ptrdiff_t LookaheadMinK = 256;
    //  above LookaheadMaxCols columns of k per thread the recursive
    //  variant (its trailing updates on the frame share the packed blocks)
    static constexpr std::ptrdiff_t LookaheadMaxCols = 1000;
    //  OPENMP=1: the task graph from TasksMinThreads threads on
    static constexpr std::ptrdiff_t TasksMinThreads = 9;
    //  panel widths of the lookahead: adaptive between NBMin and NB
    static constexpr std::ptrdiff_t NB = 256;
    static constexpr std::ptrdiff_t NBMin = 64;
};

template <typename T> struct GetrfParam<std::complex<T>> : GetrfParam<T>
{
    static constexpr std::ptrdiff_t NB = 128;
    static constexpr std::ptrdiff_t NBMin = 32;
};

template <typename T>
GetrfPlan
getrfSelect(std::ptrdiff_t m, std::ptrdiff_t n)
{
    typedef GetrfParam<T> P;

    const std::ptrdiff_t k = std::min(m, n);
    const std::ptrdiff_t pool = ThreadPool::instance().size();
    const std::ptrdiff_t byK = k / P::MinCols;
    GetrfPlan plan;
    plan.threads = byK < 1 ? 1 : (byK < pool ? byK : pool);
    plan.variant = GetrfPlan::Recursive;
    plan.nb = P::NB;
    plan.nbMin = P::NBMin;
    plan.chunk = 0;
    if (plan.threads > 1 && k >= P::LookaheadMinK &&
        k <= P::LookaheadMaxCols * plan.threads) {
	plan.variant = GetrfPlan::Lookahead;
    }
#ifdef _OPENMP
    //  OPENMP=1, many threads: the task graph (getrf.tcc decides with its
    //  own parameters)
    if (plan.threads >= P::TasksMinThreads) {
	plan.variant = GetrfPlan::Tasks;
    }
#endif
#ifdef ULMBLAS_GETRF_STUDY
    //  benchmarks of the variants without rebuilding
    if (const char *v = std::getenv("GETRF_VARIANT")) {
	plan.variant = v[0] == 'l'   ? GetrfPlan::Lookahead
	               : v[0] == 't' ? GetrfPlan::Tasks
	                             : GetrfPlan::Recursive;
    }
    if (const char *v = std::getenv("GETRF_NB")) {
	plan.nb = std::atol(v);
    }
    if (const char *v = std::getenv("GETRF_NBMIN")) {
	plan.nbMin = std::atol(v);
    }
    if (const char *v = std::getenv("GETRF_CHUNK")) {
	plan.chunk = std::atol(v);
    }
    if (const char *v = std::getenv("GETRF_THREADS")) {
	plan.threads = std::atol(v);
    }
#endif
    return plan;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRFSELECT_H
