#ifndef ULMBLAS_LEVEL3_GEMM_THREADS_H
#define ULMBLAS_LEVEL3_GEMM_THREADS_H 1

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <ulmblas/auxiliary/threadpool.h>
#include <ulmblas/config/tuned.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ulmBLAS {

//
//  The number of threads of gemm and their partition, with the parameters
//  of tuned() (ulmblas/config/tuned.h: measured by make tune, otherwise the
//  values below).
//
//  forProblem: the decision by the work m n k that make tune measured
//  (tuned().gemmThreads), at most the threads of the pool
//  (ULMBLAS_NUM_THREADS, see ulmblas/auxiliary/threadpool.h); where it has
//  none (by default everywhere) as OpenBLAS 0.3.34 (interface/gemm.c), the
//  threads of the pool, fewer for small problems: at least MinWork
//  multiply-adds per thread
//  (tuned().gemmMinWork; OpenBLAS: 65536 * 4 real, 8192 * 4 complex), one
//  thread below 2 MinWork; one thread with SME
//  (USE_SME) and inside a parallel region of OpenMP (OPENMP=1).  The kernels
//  for small matrices are single threaded: they leave problems with more than
//  one thread to their fallback, the frame algorithm.
//
//  partition: as BLIS 2.0 (bli_rntm_factorize, bli_thread_partition_2x2):
//  nt = jcWays icWays jrWays.  nt is split into icWays x jc ways with the
//  ratio of the work m RatioM : n RatioN (prime factors in increasing order
//  to the side with more work per thread, then a factor 2 moved if that
//  balances better; fewer than 4 threads all on the larger side); jrWays
//  is the largest divisor of the jc ways up to MaxJR, jcWays the rest.
//  RatioM, RatioN and MaxJR: tuned().gemmRatioM, gemmRatioN, gemmMaxJR (the
//  constants of BLIS's configuration for the platform: skx (Ice Lake) 3:2
//  and MaxJR 4, zen3 (Zen 4; BLIS 2.0 has no zen4 configuration) 1:1 and
//  MaxJR 1, otherwise 1:1 and MaxJR 4).  At most one panel of MR rows
//  (NR columns) per group of rows (columns).
//
//  Integer arithmetic only: a level 3 call with exact operands raises no
//  floating-point exception flags (T0359; a conversion of a double to an
//  integer raises inexact, also on a branch whose result is not used).
//
struct GemmThreads
{
    template <typename T, typename IndexType>
    static std::ptrdiff_t
    forProblem(IndexType m, IndexType n, IndexType k)
    {
#if defined(USE_SME)
	//  the performance cores share the SME unit: no gain from threads
	//  (Apple M4, M5) until measurements show otherwise
	return 1;
#endif
#if defined(_OPENMP)
	//  inside a parallel region of OpenMP (OPENMP=1: the tasks of the
	//  LAPACK functions call gemm): the threads are OpenMP's
	if (omp_in_parallel()) {
	    return 1;
	}
#endif
	const std::ptrdiff_t pool = ThreadPool::instance().size();
	if (pool <= 1) {
	    return 1;
	}
	const Tuned &p = tuned();
	const std::ptrdiff_t t = TunedIndex<T>::value;
	const std::int64_t m_ = std::int64_t(m), n_ = std::int64_t(n),
	                   k_ = std::int64_t(k);
	if (m_ <= 0 || n_ <= 0 || k_ <= 0) {
	    return 1;
	}
	//  the decision by the work m n k (make tune): v threads, 0 all of
	//  the pool, -1 by MinWork
	const std::int64_t v = p.gemmThreads[t](product(product(m_, n_), k_));
	if (v >= 0) {
	    return v == 0 || v >= std::int64_t(pool) ? pool : std::ptrdiff_t(v);
	}
	//  m n k / MinWork, at most pool: an operand above pool MinWork (or
	//  m n above) gives pool, otherwise no overflow
	const std::int64_t w0 = p.gemmMinWork[t];
	const std::int64_t w = w0 > 0 ? w0 : 1;
	const std::int64_t limit = std::int64_t(pool) * w;
	if (m_ > limit || n_ > limit || k_ > limit || m_ * n_ > limit) {
	    return pool;
	}
	const std::int64_t byWork = m_ * n_ * k_ / w;
	return byWork < 2 ? 1 : (byWork < pool ? std::ptrdiff_t(byWork) : pool);
    }

    //  a b for a, b > 0, the largest value on overflow
    static std::int64_t
    product(std::int64_t a, std::int64_t b)
    {
	const std::int64_t max = std::numeric_limits<std::int64_t>::max();
	return a > max / b ? max : a * b;
    }

    //  BLIS's bli_thread_partition_2x2_fast: nt = nt1 nt2, nt1 : nt2 about
    //  work1 : work2.  BLIS divides the work by the factors (doubles); here
    //  work1 / t1 and work2 / t2 are compared as work1 t2 and work2 t1.
    static void
    partition2x2(std::ptrdiff_t nt, std::int64_t work1, std::int64_t work2,
                 std::ptrdiff_t &nt1, std::ptrdiff_t &nt2)
    {
	if (nt < 4) {
	    nt1 = work1 >= work2 ? nt : 1;
	    nt2 = work1 < work2 ? nt : 1;
	    return;
	}
	std::int64_t t1 = 1, t2 = 1;
	std::ptrdiff_t rest = nt;
	for (std::ptrdiff_t f = 2; rest > 1;) {
	    if (rest % f != 0) {
		++f;
		continue;
	    }
	    rest /= f;
	    if (work1 * t2 > work2 * t1) {
		t1 *= f;
	    } else {
		t2 *= f;
	    }
	}
	//  a = work1 / t1, b = work2 / t2; times 2 t1 t2:
	//  a > b:  |a/2 - 2b| < a - b,  a < b:  |b/2 - 2a| < b - a
	const std::int64_t a = work1 * t2, b = work2 * t1;
	if (a > b && t2 % 2 == 0) {
	    if (std::llabs(a - 4 * b) < 2 * (a - b)) {
		t1 *= 2;
		t2 /= 2;
	    }
	} else if (a < b && t1 % 2 == 0) {
	    if (std::llabs(b - 4 * a) < 2 * (b - a)) {
		t1 /= 2;
		t2 *= 2;
	    }
	}
	nt1 = std::ptrdiff_t(t1);
	nt2 = std::ptrdiff_t(t2);
    }

    struct Constants
    {
	std::int64_t ratioM, ratioN;
	std::ptrdiff_t maxJR;
    };

    static Constants
    constants()
    {
	const Tuned &p = tuned();
	return Constants{p.gemmRatioM > 0 ? p.gemmRatioM : 1,
	                 p.gemmRatioN > 0 ? p.gemmRatioN : 1,
	                 p.gemmMaxJR > 0 ? p.gemmMaxJR : 1};
    }

    static void
    partition(std::ptrdiff_t nt, std::ptrdiff_t m, std::ptrdiff_t n,
              std::ptrdiff_t MR, std::ptrdiff_t NR, std::ptrdiff_t &jcWays,
              std::ptrdiff_t &icWays, std::ptrdiff_t &jrWays)
    {
	const Constants c = constants();
	std::ptrdiff_t ic, jc;
	partition2x2(nt, std::int64_t(m) * c.ratioM, std::int64_t(n) * c.ratioN,
	             ic, jc);
	std::ptrdiff_t jr = 1;
	for (std::ptrdiff_t r = c.maxJR; r > 1; --r) {
	    if (jc % r == 0) {
		jr = r;
		break;
	    }
	}
	jcWays = jc / jr;
	icWays = ic;
	jrWays = jr;
	//  at least one panel per group (fewer groups otherwise; the threads
	//  then compute nothing in that loop)
	const std::ptrdiff_t mp = (m + MR - 1) / MR, np = (n + NR - 1) / NR;
	if (icWays > mp) {
	    icWays = mp;
	}
	if (jcWays * jrWays > np) {
	    jcWays = np / jrWays > 0 ? np / jrWays : 1;
	    if (jcWays * jrWays > np) {
		jrWays = np;
	    }
	}
    }

    //  OpenBLAS's partition (gemm_thread of level3_thread.c): nthreadsM
    //  threads for ranges of rows with at least switchRatio rows each,
    //  nthreadsN for ranges of columns; then nthreadsM / div, nthreadsN div
    //  for the divisor div of nthreadsM that minimizes n nthreadsM +
    //  m nthreadsN.  switchRatio: 16 for float, 8 for double (SKYLAKEX).
    static void
    partitionOpenBLAS(std::ptrdiff_t nt, std::ptrdiff_t m, std::ptrdiff_t n,
                      std::ptrdiff_t switchRatio, std::ptrdiff_t &nthreadsN,
                      std::ptrdiff_t &nthreadsM)
    {
	if (m < 2 * switchRatio) {
	    nthreadsM = 1;
	} else {
	    nthreadsM = nt;
	    while (m < nthreadsM * switchRatio) {
		nthreadsM /= 2;
	    }
	}
	if (n < switchRatio * nthreadsM) {
	    nthreadsN = 1;
	    return;
	}
	nthreadsN =
	    (n + switchRatio * nthreadsM - 1) / (switchRatio * nthreadsM);
	if (nthreadsM * nthreadsN > nt) {
	    nthreadsN = nt / nthreadsM;
	}
	std::ptrdiff_t cost = 0, div = 0;
	for (std::ptrdiff_t i = 1; i * i <= nthreadsM; ++i) {
	    if (nthreadsM % i) {
		continue;
	    }
	    const std::ptrdiff_t j = nthreadsM / i;
	    const std::ptrdiff_t costI = n * j + m * nthreadsN * i;
	    const std::ptrdiff_t costJ = n * i + m * nthreadsN * j;
	    if (cost == 0 || costI < cost) {
		cost = costI;
		div = i;
	    }
	    if (costJ < cost) {
		cost = costJ;
		div = j;
	    }
	}
	if (div > 1) {
	    nthreadsM /= div;
	    nthreadsN *= div;
	}
    }
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_THREADS_H
