//
//  Measurement program of the parameter tuning (make tune-params, see
//  tune/README.md): measures the candidates of a group of parameters of
//  ulmblas/config/tuned.h on this machine.
//
//      measure info                       platform, build, threads of the pool
//      measure param TYPE                 the parameters for TYPE: the
//                                         defaults and those of the build
//      measure count GROUP TYPE           the number of problems of a group
//      measure run GROUP TYPE ROUNDS FIRST COUNT
//                                         measures the problems FIRST to
//                                         FIRST+COUNT-1 with all settings
//      measure verify GROUP TYPE ROUNDS FIRST COUNT
//                                         ... with the defaults written by
//                                         hand and the tuned parameters
//
//  TYPE is s, d, c or z.  A setting is a value of the parameters (tuned()),
//  a number of threads of the pool and a variant of trsm and trmm forced
//  (TrShared::forced).  The settings of a problem run interleaved in random
//  order, ROUNDS times; a sample is the time of enough calls for at least
//  20 ms after one call that is not timed (not for calls of 50 ms or more),
//  with the workers of the pool asleep before (no worker that still spins
//  after the previous sample on the core of a thread of this one).
//  Output, one line per problem and setting:
//
//      group type op m n k pt threads setting GFLOP/s (median, min, max)
//      MR NR KC1
//
//  pt: the threads of the problem (1, or 0 for all of the pool); threads:
//  the threads the problem runs on with this setting as far as the
//  measurement program knows it (gemm, trsm, trmm: GemmThreads::forProblem;
//  level 2: the pool); MR, NR, KC1: of the frame that the selection table
//  takes for trsm and trmm (KC1: KCmax in whole panels), 0 otherwise.  The
//  setting "current" is the parameters of the build (the defaults or the
//  values of an earlier make tune).
//
//  Groups:
//
//      gemm-threads    gemm on 1, 2, 3, 4, 6, ... threads and on all
//                      numbers of threads that a candidate of gemmMinWork
//                      gives
//      gemm-partition  gemm on all threads with candidates of gemmRatioM,
//                      gemmRatioN, gemmMaxJR
//      trsm-kb         trsm (A on the left, lower) on one and all threads
//                      with candidates of kb and trsmInvert
//      trmm-kb         trmm on one and all threads, candidates of kb
//      trsm-variant    trsm on all threads: the shared variant and the
//                      columns variant with 1, 2, 4, 8 and 16 panels per
//                      thread at least
//      trmm-variant    ... trmm
//      level2          gemv (A column major and transposed) and ger on 1,
//                      2, ... threads
//      potrf           potrf (lower) with OpenMP (OPENMP=1): serial and as
//                      tasks with candidates of the tile size
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <ulmblas/ulmblas.h>
#include <ulmblas/config/platform.h>
#include <ulmblas/config/tuned.h>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/level3/gemm/trshared.h>

namespace {

typedef std::ptrdiff_t Index;
using ulmBLAS::Tuned;

enum Op
{
    OpGemm,
    OpTrsm,
    OpTrmm,
    OpGemvN,
    OpGemvT,
    OpGer,
    OpPotrf
};

const char *
opName(Op op)
{
    switch (op) {
    case OpGemm:
	return "gemm";
    case OpTrsm:
	return "trsm";
    case OpTrmm:
	return "trmm";
    case OpGemvN:
	return "gemv-n";
    case OpGemvT:
	return "gemv-t";
    case OpPotrf:
	return "potrf";
    default:
	return "ger";
    }
}

//  a problem; threads: 0 all threads of the pool
struct Problem
{
    Op op;
    Index m, n, k, threads;
};

//  a setting: the parameters, the threads of the pool (0: the problem's),
//  the variant of trsm and trmm (-1: chosen by the parameters)
struct Setting
{
    std::string name;
    Tuned param;
    Index threads;
    Index forced;
};

Index
poolSize()
{
    return ulmBLAS::ThreadPool::instance().size();
}

//  the size of the pool by default (before any setSize)
Index
defaultPool()
{
    static const Index value = poolSize();
    return value;
}

//
//  The build: the condition under which the parameters measured here
//  apply (the instruction set of the kernels), written into the generated
//  header.
//
const char *
buildCondition()
{
#if defined(USE_SME)
    return "defined(USE_SME)";
#elif defined(__AVX512F__)
    return "defined(__AVX512F__) && defined(__x86_64__)";
#elif defined(__AVX2__) && defined(__FMA__)
    return "defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512F__) "
           "&& defined(__x86_64__)";
#elif defined(__aarch64__)
    return "defined(__aarch64__) && !defined(USE_SME)";
#elif defined(__x86_64__)
    return "defined(__x86_64__) && !defined(__AVX2__)";
#else
    return "1";
#endif
}

//  SME: level 3 on one thread (GemmThreads::forProblem)
bool
smeBuild()
{
#if defined(USE_SME)
    return true;
#else
    return false;
#endif
}

//
//  The problems of a group (none: nothing to measure on this machine)
//
std::vector<Problem>
problems(const std::string &group)
{
    std::vector<Problem> p;
    if (group == "gemm-threads") {
	const Index sizes[] = {16,  24,  32,  48,  64,  96,
	                       128, 192, 256, 384, 512, 768};
	if (defaultPool() <= 1 || smeBuild()) {
	    return p;
	}
	for (Index s : sizes) {
	    p.push_back({OpGemm, s, s, s, 0});
	}
	const Index rect[][3] = {
	    {256, 256, 16},  {512, 512, 32},  {1024, 1024, 32}, {64, 64, 1024},
	    {32, 1024, 256}, {1024, 32, 256}, {2000, 64, 64},   {64, 2000, 64}};
	for (const Index *r : rect) {
	    p.push_back({OpGemm, r[0], r[1], r[2], 0});
	}
    } else if (group == "gemm-partition") {
	if (defaultPool() <= 1 || smeBuild()) {
	    return p;
	}
	const Index shapes[][3] = {{1000, 1000, 1000}, {2000, 2000, 2000},
	                           {4000, 4000, 4000}, {4000, 500, 1000},
	                           {500, 4000, 1000},  {2000, 2000, 256},
	                           {8000, 200, 200},   {200, 8000, 200}};
	for (const Index *s : shapes) {
	    p.push_back({OpGemm, s[0], s[1], s[2], 0});
	}
    } else if (group == "trsm-kb" || group == "trmm-kb") {
	const Op op = group == "trsm-kb" ? OpTrsm : OpTrmm;
	const Index ms[] = {200, 500, 1000, 2000};
	const Index ns[] = {200, 1000, 3000};
	const Index threads[] = {1, 0};
	for (Index t : threads) {
	    //  all threads: not with SME (one thread, GemmThreads)
	    if (t == 0 && (defaultPool() <= 1 || smeBuild())) {
		continue;
	    }
	    for (Index m : ms) {
		for (Index n : ns) {
		    p.push_back({op, m, n, m, t});
		}
	    }
	}
    } else if (group == "trsm-variant" || group == "trmm-variant") {
	const Op op = group == "trsm-variant" ? OpTrsm : OpTrmm;
	const Index ms[] = {250, 500, 1000, 2000, 4000};
	const Index ns[] = {64, 250, 500, 1000, 2000, 4000};
	if (defaultPool() > 1 && !smeBuild()) {
	    for (Index m : ms) {
		for (Index n : ns) {
		    p.push_back({op, m, n, m, 0});
		}
	    }
	}
    } else if (group == "potrf") {
#if defined(_OPENMP)
	const Index sizes[] = {64,  100,  150,  200,  300, 500,
	                       800, 1200, 2000, 3000, 5000};
	if (ulmBLAS::numThreads() > 1) {
	    for (Index s : sizes) {
		p.push_back({OpPotrf, s, s, s, 0});
	    }
	}
#endif
    } else if (group == "level2") {
	const Index sizes[] = {32,  48,  64,  96,   128,  192,  256,
	                       384, 512, 768, 1024, 1536, 2048, 4096};
	const Op ops[] = {OpGemvN, OpGemvT, OpGer};
	for (Op op : ops) {
	    for (Index s : sizes) {
		p.push_back({op, s, s, 1, 0});
	    }
	}
    }
    return p;
}

//  the numbers of threads to measure: 1, 2, 3, 4, 6, 8, 12, ... and the
//  pool
std::vector<Index>
threadCounts()
{
    std::vector<Index> t;
    const Index pool = defaultPool();
    for (Index a = 1; a <= pool; a *= 2) {
	t.push_back(a);
	if (3 * a / 2 > a && 3 * a / 2 <= pool && 3 * a / 2 != 2 * a) {
	    t.push_back(3 * a / 2);
	}
    }
    t.push_back(pool);
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    return t;
}

//  the candidates of gemmMinWork (make tune: tune/params/params.py)
const std::int64_t minWorkCandidates[] = {
    1 << 12, 1 << 13, 1 << 14, 3 << 13, 1 << 15, 3 << 14,
    1 << 16, 3 << 15, 1 << 17, 3 << 16, 1 << 18, 3 << 17,
    1 << 19, 3 << 18, 1 << 20, 3 << 19, 1 << 21, 1 << 22};

//  GemmThreads::forProblem with minimal work w on a pool of p threads
Index
threadsByWork(Index m, Index n, Index k, std::int64_t w, Index p)
{
    const std::int64_t byWork =
        std::int64_t(m) * std::int64_t(n) * std::int64_t(k) / w;
    return byWork < 2 ? 1 : (byWork < p ? Index(byWork) : p);
}

//
//  MR, NR and KCmax (whole panels) of the frame that the selection table
//  takes for trsm, as a plan of the table (as TrsmOp, gemm/frametrsm.h)
//
struct FrameInfo
{
    Index MR, NR, KC1;
};

template <typename T> struct InfoOp
{
    typedef FrameInfo Plan;

    static constexpr std::size_t sizeC = sizeof(T);
    static constexpr std::size_t sizeA = sizeof(T);

    template <typename F>
    static Plan
    plan(const char *name)
    {
	(void)name;
	typedef typename ulmBLAS::GemmFrameOf<F>::type Frame;
	typedef typename Frame::BackendType Backend;
	typedef typename Frame::BlocksType Blocks;
	return Plan{Backend::MR, Backend::NR,
	            Blocks::KCmax / Backend::MR * Backend::MR};
    }
};

//
//  The settings of a group for a problem of type index t
//
std::vector<Setting>
settings(const std::string &group, std::ptrdiff_t t, const Problem &problem)
{
    using namespace ulmBLAS;
    const Tuned base = tuned();
    std::vector<Setting> s;
    char name[64];
    if (group == "gemm-threads") {
	//  the numbers of threads of the candidates of gemmMinWork (and of
	//  the default and the tuned value)
	std::vector<Index> nts;
	for (std::int64_t w : minWorkCandidates) {
	    nts.push_back(threadsByWork(problem.m, problem.n, problem.k, w,
	                                defaultPool()));
	}
	const std::int64_t own[] = {tunedDefaults(platform()).gemmMinWork[t],
	                            base.gemmMinWork[t]};
	for (std::int64_t w : own) {
	    nts.push_back(threadsByWork(problem.m, problem.n, problem.k, w,
	                                defaultPool()));
	}
	//  and 1, 2, 3, 4, 6, ... (gemmThreads)
	for (Index nt : threadCounts()) {
	    nts.push_back(nt);
	}
	std::sort(nts.begin(), nts.end());
	nts.erase(std::unique(nts.begin(), nts.end()), nts.end());
	s.push_back({"current", base, 0, -1});
	for (Index nt : nts) {
	    Tuned p = base;
	    p.gemmThreads[t] = TunedSteps{-1};
	    p.gemmMinWork[t] = 1;
	    std::snprintf(name, sizeof(name), "t%td", nt);
	    s.push_back({name, p, nt, -1});
	}
    } else if (group == "gemm-partition") {
	s.push_back({"current", base, 0, -1});
	const Index c[][3] = {{1, 1, 1}, {1, 1, 2}, {1, 1, 4}, {3, 2, 4},
	                      {3, 2, 1}, {2, 1, 4}, {1, 2, 4}, {2, 3, 4}};
	for (const Index *x : c) {
	    Tuned p = base;
	    p.gemmRatioM = x[0];
	    p.gemmRatioN = x[1];
	    p.gemmMaxJR = x[2];
	    std::snprintf(name, sizeof(name), "r%td:%td:%td", x[0], x[1], x[2]);
	    s.push_back({name, p, 0, -1});
	}
    } else if (group == "trsm-kb" || group == "trmm-kb") {
	const bool trsm = group == "trsm-kb";
	s.push_back({"current", base, problem.threads, -1});
	const Index kbs[] = {96, 128, 192, 256, 384, 512};
	for (Index kb : kbs) {
	    for (Index inv = 0; inv < (trsm ? 2 : 1); ++inv) {
		Tuned p = base;
		(trsm ? p.trsm : p.trmm).kb[t] = TunedSteps{kb};
		if (trsm) {
		    p.trsmInvert[t] = inv;
		    std::snprintf(name, sizeof(name), "kb%td-inv%td", kb, inv);
		} else {
		    std::snprintf(name, sizeof(name), "kb%td", kb);
		}
		s.push_back({name, p, problem.threads, -1});
	    }
	}
    } else if (group == "trsm-variant" || group == "trmm-variant") {
	const bool trsm = group == "trsm-variant";
	s.push_back({"current", base, 0, -1});
	s.push_back({"shared", base, 0, 1});
	const Index minp[] = {1, 2, 4, 8, 16};
	for (Index q : minp) {
	    Tuned p = base;
	    (trsm ? p.trsm : p.trmm).minPanels[t] = q;
	    std::snprintf(name, sizeof(name), "c%td", q);
	    s.push_back({name, p, 0, 0});
	}
    } else if (group == "potrf") {
	s.push_back({"current", base, 0, -1});
	Tuned serial = base;
	serial.potrfSerialN = problem.m;
	s.push_back({"serial", serial, 0, -1});
	const Index nbs[] = {32, 48, 64, 96, 128, 192, 256};
	for (Index nb : nbs) {
	    if (nb >= problem.m) {
		continue;
	    }
	    Tuned p = base;
	    p.potrfSerialN = 0;
	    p.potrfNb = TunedSteps{std::int64_t(nb)};
	    std::snprintf(name, sizeof(name), "nb%td", nb);
	    s.push_back({name, p, 0, -1});
	}
    } else if (group == "level2") {
	s.push_back({"current", base, 0, -1});
	for (Index nt : threadCounts()) {
	    Tuned p = base;
	    p.gemvNThreads[t] = TunedSteps{std::int64_t(nt)};
	    p.gemvTThreads[t] = TunedSteps{std::int64_t(nt)};
	    p.gerThreads[t] = TunedSteps{std::int64_t(nt)};
	    std::snprintf(name, sizeof(name), "t%td", nt);
	    s.push_back({name, p, 0, -1});
	}
    }
    return s;
}

//
//  Random elements in [-1, 1) (complex: real and imaginary part), the
//  floating-point operations of a multiply-add
//
template <typename T> struct Random
{
    typedef T Real;
    static constexpr double flops = 2;

    template <typename Gen>
    static T
    value(Gen &gen)
    {
	std::uniform_real_distribution<double> dist(-1, 1);
	return T(dist(gen));
    }
};

template <typename T> struct Random<std::complex<T>>
{
    typedef T Real;
    static constexpr double flops = 8;

    template <typename Gen>
    static std::complex<T>
    value(Gen &gen)
    {
	std::uniform_real_distribution<double> dist(-1, 1);
	const T re = T(dist(gen));
	return std::complex<T>(re, T(dist(gen)));
    }
};

double
seconds()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

//  the operands of a problem
template <typename T> struct Operands
{
    std::vector<T> A, A0, B, C, x, y;

    Operands(const Problem &p, std::mt19937 &gen)
    {
	Index sa = 0, sb = 0, sc = 0;
	switch (p.op) {
	case OpGemm:
	    sa = p.m * p.k, sb = p.k * p.n, sc = p.m * p.n;
	    break;
	case OpTrsm:
	case OpTrmm:
	    sa = p.m * p.m, sb = p.m * p.n;
	    break;
	default:
	    sa = p.m * p.n;
	    break;
	}
	A.resize(std::size_t(sa));
	B.resize(std::size_t(sb));
	C.resize(std::size_t(sc));
	x.resize(std::size_t(std::max(p.m, p.n)));
	y.resize(std::size_t(std::max(p.m, p.n)));
	std::vector<T> *all[] = {&A, &B, &C, &x, &y};
	for (std::vector<T> *v : all) {
	    for (T &e : *v) {
		e = Random<T>::value(gen);
	    }
	}
	//  trsm: a well conditioned triangle (diagonal m), B stays bounded
	//  by solving and multiplying alternately (trmm, trsm: scaled)
	if (p.op == OpTrsm || p.op == OpTrmm) {
	    for (Index i = 0; i < p.m; ++i) {
		A[std::size_t(i * p.m + i)] = T(double(p.m));
	    }
	}
	//  potrf: A0 Hermitian with a dominant diagonal (positive definite),
	//  copied to A before each call
	if (p.op == OpPotrf) {
	    for (Index j = 0; j < p.m; ++j) {
		for (Index i = 0; i < j; ++i) {
		    A[std::size_t(j * p.m + i)] =
		        conjugate(A[std::size_t(i * p.m + j)]);
		}
		A[std::size_t(j * p.m + j)] = T(2 * double(p.m));
	    }
	    A0 = A;
	}
    }

    template <typename R>
    static R
    conjugate(const R &x)
    {
	return x;
    }

    template <typename R>
    static std::complex<R>
    conjugate(const std::complex<R> &x)
    {
	return std::conj(x);
    }
};

//  one call of the problem
template <typename T>
void
call(const Problem &p, Operands<T> &o)
{
    const Index one = 1;
    switch (p.op) {
    case OpGemm:
	ulmBLAS::gemm(p.m, p.n, p.k, T(1), false, o.A.data(), one, p.m, false,
	              o.B.data(), one, p.k, T(0.5), o.C.data(), one, p.m);
	break;
    case OpTrsm:
	ulmBLAS::trlsm(p.m, p.n, T(double(p.m)), false, false, o.A.data(), one,
	               p.m, o.B.data(), one, p.m);
	break;
    case OpTrmm:
	ulmBLAS::trlmm(p.m, p.n, T(1.0 / double(p.m)), false, false, o.A.data(),
	               one, p.m, o.B.data(), one, p.m);
	break;
    case OpGemvN:
	ulmBLAS::gemv(p.m, p.n, T(1), o.A.data(), one, p.m, o.x.data(), one,
	              T(0.5), o.y.data(), one);
	break;
    case OpGemvT:
	ulmBLAS::gemv(p.n, p.m, T(1), o.A.data(), p.m, one, o.x.data(), one,
	              T(0.5), o.y.data(), one);
	break;
    case OpGer:
	ulmBLAS::ger(p.m, p.n, T(1e-3), o.x.data(), one, o.y.data(), one,
	             o.A.data(), one, p.m);
	break;
    case OpPotrf:
	std::copy(o.A0.begin(), o.A0.end(), o.A.begin());
	ulmBLAS::potrf(true, p.m, o.A.data(), one, p.m);
	break;
    }
}

double
flops(const Problem &p)
{
    switch (p.op) {
    case OpGemm:
	return double(p.m) * double(p.n) * double(p.k);
    case OpTrsm:
    case OpTrmm:
	return double(p.m) * double(p.m) * double(p.n) / 2;
    case OpPotrf:
	return double(p.m) * double(p.m) * double(p.m) / 6;
    default:
	return double(p.m) * double(p.n);
    }
}

//  waits until the workers of the pool sleep (see ThreadPool)
void
idle()
{
    if (defaultPool() > 1) {
	const std::int64_t ns =
	    ulmBLAS::threads::idleSpinNanoseconds() + 10000000;
	std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
    }
}

//  applies a setting for problem p
void
apply(const Setting &s, const Problem &p)
{
    ulmBLAS::tuned() = s.param;
    const Index threads = s.threads > 0 ? s.threads : p.threads;
    ulmBLAS::ThreadPool::instance().setSize(threads > 0 ? threads
                                                        : defaultPool());
    ulmBLAS::TrShared::forced().store(s.forced);
}

//  the threads of problem p under the current setting (see above)
template <typename T>
Index
threadsOf(const Problem &p)
{
    if (p.op == OpTrsm || p.op == OpTrmm) {
	return ulmBLAS::GemmThreads::forProblem<T>(p.m, p.n, (p.m + 1) / 2);
    }
    if (p.op == OpGemm) {
	return ulmBLAS::GemmThreads::forProblem<T>(p.m, p.n, p.k);
    }
    return poolSize();
}

template <typename T>
int
measure(const std::string &group, char type, bool verify, Index rounds,
        Index first, Index count)
{
    using namespace ulmBLAS;
    const std::ptrdiff_t t = TunedIndex<T>::value;
    const std::vector<Problem> all = problems(group);
    const Tuned tunedValue = tuned();
    std::mt19937 gen(std::uint32_t(first + 1));

    for (Index q = first; q < first + count && q < Index(all.size()); ++q) {
	const Problem &p = all[std::size_t(q)];
	std::vector<Setting> set;
	if (verify) {
	    //  the problem's own threads, the variant by the parameters
	    set.push_back({"default", tunedDefaults(platform()), 0, -1});
	    set.push_back({"tuned", tunedValue, 0, -1});
	} else {
	    tuned() = tunedValue;
	    set = settings(group, t, p);
	}
	Operands<T> o(p, gen);
	const double f = Random<T>::flops * flops(p);
	FrameInfo info = {0, 0, 0};
	if (p.op == OpTrsm || p.op == OpTrmm) {
	    info = level3Select<T, InfoOp<T>>(p.m, p.n, p.m);
	}

	//  calls per sample: at least 20 ms (estimated with the first
	//  setting)
	apply(set[0], p);
	call(p, o);
	double t0 = seconds();
	call(p, o);
	const double once = seconds() - t0;
	const Index reps =
	    std::max(Index(1), Index(20e-3 / std::max(once, 1e-7)));

	std::vector<std::vector<double>> samples(set.size());
	std::vector<Index> threads(set.size());
	std::vector<std::size_t> order(set.size());
	for (std::size_t c = 0; c < set.size(); ++c) {
	    order[c] = c;
	}
	for (Index r = 0; r < rounds; ++r) {
	    std::shuffle(order.begin(), order.end(), gen);
	    for (std::size_t c : order) {
		//  the workers of the pool asleep (they spin for a while after
		//  a region, on the cores of other threads)
		idle();
		apply(set[c], p);
		threads[c] = threadsOf<T>(p);
		if (once < 50e-3) {
		    call(p, o);
		}
		double s = seconds();
		for (Index k = 0; k < reps; ++k) {
		    call(p, o);
		}
		s = (seconds() - s) / double(reps);
		samples[c].push_back(f / s * 1e-9);
	    }
	}
	for (std::size_t c = 0; c < set.size(); ++c) {
	    std::vector<double> &x = samples[c];
	    std::sort(x.begin(), x.end());
	    std::printf("%s\t%c\t%s\t%td\t%td\t%td\t%td\t%td\t%s\t%.3f\t%.3f"
	                "\t%.3f\t%td\t%td\t%td\n",
	                group.c_str(), type, opName(p.op), p.m, p.n, p.k,
	                p.threads, threads[c], set[c].name.c_str(),
	                x[x.size() / 2], x.front(), x.back(), info.MR, info.NR,
	                info.KC1);
	}
	std::fflush(stdout);
    }
    tuned() = tunedValue;
    ThreadPool::instance().setSize(0);
    TrShared::forced().store(-1);
    return 0;
}

//  the steps as v0,b0,v1,...
std::string
stepsText(const ulmBLAS::TunedSteps &steps)
{
    std::string text = std::to_string(steps.value[0]);
    for (std::ptrdiff_t i = 0; i < steps.count; ++i) {
	text += "," + std::to_string(steps.bound[i]) + "," +
	        std::to_string(steps.value[i + 1]);
    }
    return text;
}

//  the parameters of type index t (one line, name=value)
void
printParam(const char *label, const Tuned &p, std::ptrdiff_t t)
{
    std::printf("%s\tgemmThreads=%s\tgemmMinWork=%lld\tgemmRatioM=%lld"
                "\tgemmRatioN=%lld\tgemmMaxJR=%td",
                label, stepsText(p.gemmThreads[t]).c_str(),
                (long long)p.gemmMinWork[t], (long long)p.gemmRatioM,
                (long long)p.gemmRatioN, p.gemmMaxJR);
    const Tuned::Tr *ops[] = {&p.trsm, &p.trmm};
    const char *names[] = {"trsm", "trmm"};
    for (std::size_t i = 0; i < 2; ++i) {
	std::printf("\t%s.kb=%s\t%s.sharedBlocks=%td\t%s.sharedPanels=%td"
	            "\t%s.minPanels=%td",
	            names[i], stepsText(ops[i]->kb[t]).c_str(), names[i],
	            ops[i]->sharedBlocks[t], names[i], ops[i]->sharedPanels[t],
	            names[i], ops[i]->minPanels[t]);
    }
    std::printf("\ttrsmInvert=%td\tgemvNThreads=%s\tgemvTThreads=%s"
                "\tgerThreads=%s\tpotrfNb=%s\tpotrfSerialN=%td\n",
                p.trsmInvert[t], stepsText(p.gemvNThreads[t]).c_str(),
                stepsText(p.gemvTThreads[t]).c_str(),
                stepsText(p.gerThreads[t]).c_str(),
                stepsText(p.potrfNb).c_str(), p.potrfSerialN);
}

int
dispatch(const std::string &group, char type, bool verify, Index rounds,
         Index first, Index count)
{
    switch (type) {
    case 's':
	return measure<float>(group, type, verify, rounds, first, count);
    case 'd':
	return measure<double>(group, type, verify, rounds, first, count);
    case 'c':
	return measure<std::complex<float>>(group, type, verify, rounds, first,
	                                    count);
    case 'z':
	return measure<std::complex<double>>(group, type, verify, rounds, first,
	                                     count);
    default:
	std::fprintf(stderr, "measure: unknown type %c\n", type);
	return 2;
    }
}

} // namespace

int
main(int argc, char **argv)
{
    defaultPool();
    if (argc == 2 && std::strcmp(argv[1], "info") == 0) {
	std::printf("%s\t%s\t%td\n", ulmBLAS::platformName(ulmBLAS::platform()),
	            buildCondition(), defaultPool());
	return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "param") == 0) {
	const char types[] = "sdcz";
	const char *c = std::strchr(types, argv[2][0]);
	const std::ptrdiff_t t = c ? c - types : 1;
	printParam("default", ulmBLAS::tunedDefaults(ulmBLAS::platform()), t);
	printParam("current", ulmBLAS::tuned(), t);
	return 0;
    }
    if (argc == 4 && std::strcmp(argv[1], "count") == 0) {
	std::printf("%zu\n", problems(argv[2]).size());
	return 0;
    }
    if (argc == 7 && (std::strcmp(argv[1], "run") == 0 ||
                      std::strcmp(argv[1], "verify") == 0)) {
	return dispatch(argv[2], argv[3][0],
	                std::strcmp(argv[1], "verify") == 0,
	                Index(std::atol(argv[4])), Index(std::atol(argv[5])),
	                Index(std::atol(argv[6])));
    }
    std::fprintf(stderr,
                 "usage: measure info | param TYPE | count GROUP TYPE | "
                 "run|verify GROUP TYPE ROUNDS FIRST COUNT\n");
    return 2;
}
