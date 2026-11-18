#ifndef ULMBLAS_LEVEL2_THREADS_H
#define ULMBLAS_LEVEL2_THREADS_H 1

#include <cstddef>
#include <cstdint>
#include <ulmblas/auxiliary/threadpool.h>
#include <ulmblas/config/tuned.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ulmBLAS {

//
//  Threads of the level 2 operations (gemv, ger) on the pool of ulmBLAS
//  (ulmblas/auxiliary/threadpool.h): the number of threads for a problem
//  by the work m n (tuned().gemvNThreads, ..., ulmblas/config/tuned.h:
//  measured by make tune, by default as OpenBLAS 0.3.34), one thread
//  inside a parallel region of OpenMP (OPENMP=1, the tasks of LAPACK).
//  The work is split into parts of whole units (e.g. 64 bytes of y, so that
//  no two threads write to the same cache line), no reductions.
//
struct Level2Threads
{
    static std::ptrdiff_t
    pool()
    {
#if defined(_OPENMP)
	if (omp_in_parallel()) {
	    return 1;
	}
#endif
	return ThreadPool::instance().size();
    }

    //  the number of threads for m x n with the thresholds of T, at most
    //  one per unit of the split dimension (parts of 'units' units)
    //  columnMajor: A column major (BLAS N), otherwise row major
    template <typename T>
    static std::ptrdiff_t
    gemv(std::int64_t m, std::int64_t n, std::int64_t units, bool columnMajor)
    {
	const Tuned &p = tuned();
	const std::ptrdiff_t t = TunedIndex<T>::value;
	return threads(columnMajor ? p.gemvNThreads[t] : p.gemvTThreads[t], m,
	               n, units);
    }

    template <typename T>
    static std::ptrdiff_t
    ger(std::int64_t m, std::int64_t n, std::int64_t units)
    {
	return threads(tuned().gerThreads[TunedIndex<T>::value], m, n, units);
    }

    //  part tid of nt of [0, n) in whole units of u elements
    template <typename IndexType>
    static void
    range(std::ptrdiff_t tid, std::ptrdiff_t nt, IndexType n, IndexType u,
          IndexType &i0, IndexType &i1)
    {
	const std::int64_t units = (std::int64_t(n) + u - 1) / u;
	const std::int64_t b0 = units * tid / nt;
	const std::int64_t b1 = units * (tid + 1) / nt;
	i0 = IndexType(b0) * u;
	i1 = IndexType(b1) * u;
	if (i0 > n) {
	    i0 = n;
	}
	if (i1 > n) {
	    i1 = n;
	}
    }

  private:
    //  steps(m n) threads, 0: all of the pool; at most the pool
    static std::ptrdiff_t
    threads(const TunedSteps &steps, std::int64_t m, std::int64_t n,
            std::int64_t units)
    {
	if (m <= 0 || n <= 0) {
	    return 1;
	}
	const std::int64_t v = steps(m * n);
	if (v == 1) {
	    return 1;
	}
	const std::ptrdiff_t p = pool();
	return limit(v <= 0 || v > std::int64_t(p) ? p : std::ptrdiff_t(v),
	             units);
    }

    static std::ptrdiff_t
    limit(std::ptrdiff_t nt, std::int64_t units)
    {
	if (units < std::int64_t(nt)) {
	    nt = std::ptrdiff_t(units);
	}
	return nt < 1 ? 1 : nt;
    }
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_THREADS_H
