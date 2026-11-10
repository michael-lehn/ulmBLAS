//
//  Benchmark of ?getrf / ?potrf for s, d, c, z: the templates of ulmBLAS
//  (ulmblas/lapack, BENCH_ULMBLAS, compiled with OpenMP) or the LAPACK
//  routines of OpenBLAS (BENCH_OPENBLAS) or oneMKL (BENCH_MKL, linked with
//  mkl_rt as in Julia's MKL.jl).  Same timing as bench.cc.
//
//      bench-core-ulmblas  s|d|c|z getrf|potrf N0 N1 STEP [options]
//      bench-core-openblas s|d|c|z getrf|potrf N0 N1 STEP [options]
//      bench-core-mkl      s|d|c|z getrf|potrf N0 N1 STEP [options]
//
//  options:  --min-time S     sample until S seconds are spent (0.3)
//            --min-samples K  at least K samples (7)
//            --max-samples K  at most K samples (300)
//            --warmup K       untimed calls before sampling (2)
//
//  Matrices: m = n, ldA = n, column major.  getrf: entries with real and
//  imaginary part uniform in [-1,1]; potrf: Hermitian, n added to the
//  diagonal, lower triangle ('L').  A sample is the time of one call on a
//  fresh copy (memcpy before the clock starts).  The result of the first
//  timed call is checked with the residual of the factorization, computed
//  with zgemm of the linked library (O(n^3), once per n).
//
//  Output (one line per n):
//      prec routine lib n threads median min max samples gflops check
//      residual
//  gflops: 2/3 n^3 (getrf), 1/3 n^3 (potrf) real operations, times 4 for
//  complex (as LAPACK's flop counts).
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#ifdef BENCH_ULMBLAS
#include <omp.h>
#include <ulmblas/ulmblas.h>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/potrf.h>
#define LIBNAME "ulmblas"
#elif defined(BENCH_OPENBLAS)
#define LIBNAME "openblas"
#elif defined(BENCH_MKL)
#define LIBNAME "mkl"
#else
#error "BENCH_ULMBLAS, BENCH_OPENBLAS or BENCH_MKL"
#endif

typedef std::complex<float> cfloat;
typedef std::complex<double> cdouble;

extern "C" {
void
sgetrf_(const int *, const int *, float *, const int *, int *, int *);
void
dgetrf_(const int *, const int *, double *, const int *, int *, int *);
void
cgetrf_(const int *, const int *, cfloat *, const int *, int *, int *);
void
zgetrf_(const int *, const int *, cdouble *, const int *, int *, int *);
void
spotrf_(const char *, const int *, float *, const int *, int *, std::size_t);
void
dpotrf_(const char *, const int *, double *, const int *, int *, std::size_t);
void
cpotrf_(const char *, const int *, cfloat *, const int *, int *, std::size_t);
void
zpotrf_(const char *, const int *, cdouble *, const int *, int *, std::size_t);
void
zgemm_(const char *, const char *, const int *, const int *, const int *,
       const cdouble *, const cdouble *, const int *, const cdouble *,
       const int *, const cdouble *, cdouble *, const int *, std::size_t,
       std::size_t);
#ifdef BENCH_OPENBLAS
int
openblas_get_num_threads(void);
#endif
#ifdef BENCH_MKL
int
MKL_Get_Max_Threads(void);
#endif
}

namespace {

double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

template <typename T> struct Real
{
    typedef T type;
};
template <typename T> struct Real<std::complex<T>>
{
    typedef T type;
};
template <typename T>
bool
isComplex()
{
    return false;
}
template <>
bool
isComplex<cfloat>()
{
    return true;
}
template <>
bool
isComplex<cdouble>()
{
    return true;
}

//  conjugate, real for real types (std::conj of a real number is complex)
template <typename T>
T
cj(const T &x)
{
    return x;
}
template <typename T>
std::complex<T>
cj(const std::complex<T> &x)
{
    return std::conj(x);
}

template <typename T>
T
random(std::mt19937_64 &rng)
{
    std::uniform_real_distribution<double> u(-1, 1);
    return T(u(rng));
}

template <>
cfloat
random<cfloat>(std::mt19937_64 &rng)
{
    std::uniform_real_distribution<double> u(-1, 1);
    const float re = float(u(rng));
    return cfloat(re, float(u(rng)));
}

template <>
cdouble
random<cdouble>(std::mt19937_64 &rng)
{
    std::uniform_real_distribution<double> u(-1, 1);
    const double re = u(rng);
    return cdouble(re, u(rng));
}

#if defined(BENCH_OPENBLAS) || defined(BENCH_MKL)
int
getrf(int n, float *A, int *p)
{
    int i;
    sgetrf_(&n, &n, A, &n, p, &i);
    return i;
}
int
getrf(int n, double *A, int *p)
{
    int i;
    dgetrf_(&n, &n, A, &n, p, &i);
    return i;
}
int
getrf(int n, cfloat *A, int *p)
{
    int i;
    cgetrf_(&n, &n, A, &n, p, &i);
    return i;
}
int
getrf(int n, cdouble *A, int *p)
{
    int i;
    zgetrf_(&n, &n, A, &n, p, &i);
    return i;
}
int
potrf(int n, float *A)
{
    int i;
    spotrf_("L", &n, A, &n, &i, 1);
    return i;
}
int
potrf(int n, double *A)
{
    int i;
    dpotrf_("L", &n, A, &n, &i, 1);
    return i;
}
int
potrf(int n, cfloat *A)
{
    int i;
    cpotrf_("L", &n, A, &n, &i, 1);
    return i;
}
int
potrf(int n, cdouble *A)
{
    int i;
    zpotrf_("L", &n, A, &n, &i, 1);
    return i;
}
#else
template <typename T>
int
getrf(int n, T *A, int *p)
{
    //  1-based pivots, as LAPACK (same work as the Fortran interface)
    const long info = ulmBLAS::getrf(long(n), long(n), A, 1L, long(n), p, 1L);
    for (int i = 0; i < n; ++i) {
	++p[i];
    }
    return int(info);
}

template <typename T>
int
potrf(int n, T *A)
{
    return int(ulmBLAS::potrf(true, long(n), A, 1L, long(n)));
}
#endif

//  ||P A - L U||_F / (||A||_F n eps) or ||A - L L^H||_F / (||A||_F n eps),
//  products in complex<double> with zgemm of OpenBLAS
template <typename T>
double
residual(bool lu, int n, const std::vector<T> &A0, const std::vector<T> &F,
         const std::vector<int> &piv)
{
    const std::size_t nn = std::size_t(n) * n;
    std::vector<cdouble> L(nn, 0.0), U(nn, 0.0), C(nn, 0.0), A(nn);
    for (std::size_t q = 0; q < nn; ++q) {
	A[q] = cdouble(std::real(A0[q]), std::imag(A0[q]));
    }
    for (int j = 0; j < n; ++j) {
	for (int i = 0; i < n; ++i) {
	    const cdouble f(std::real(F[i + std::size_t(j) * n]),
	                    std::imag(F[i + std::size_t(j) * n]));
	    if (lu) {
		if (i > j)
		    L[i + std::size_t(j) * n] = f;
		if (i == j)
		    L[i + std::size_t(j) * n] = 1.0;
		if (i <= j)
		    U[i + std::size_t(j) * n] = f;
	    } else if (i >= j) {
		L[i + std::size_t(j) * n] = f;
		U[j + std::size_t(i) * n] = std::conj(f);
	    }
	}
    }
    const cdouble one(1), zero(0);
    zgemm_("N", "N", &n, &n, &n, &one, L.data(), &n, U.data(), &n, &zero,
           C.data(), &n, 1, 1);
    if (lu) {
	for (int i = 0; i < n; ++i) {
	    const int p = piv[i] - 1;
	    if (p != i) {
		for (int j = 0; j < n; ++j) {
		    std::swap(A[i + std::size_t(j) * n],
		              A[p + std::size_t(j) * n]);
		}
	    }
	}
    }
    double d = 0, a = 0;
    for (std::size_t q = 0; q < nn; ++q) {
	d += std::norm(C[q] - A[q]);
	a += std::norm(A[q]);
    }
    const double eps = std::numeric_limits<typename Real<T>::type>::epsilon();
    return std::sqrt(d) / (std::sqrt(a) * n * eps);
}

struct Options
{
    double minTime = 0.3;
    int minSamples = 7;
    int maxSamples = 300;
    int warmup = 2;
};

template <typename T>
void
run(const char *prec, bool lu, int n, const Options &opt)
{
    std::mt19937_64 rng(4711 + n);
    const std::size_t nn = std::size_t(n) * n;
    std::vector<T> A0(nn), A(nn);
    std::vector<int> piv(n);
    for (std::size_t q = 0; q < nn; ++q) {
	A0[q] = random<T>(rng);
    }
    if (!lu) {
	for (int j = 0; j < n; ++j) {
	    for (int i = j + 1; i < n; ++i) {
		A0[j + std::size_t(i) * n] = cj(A0[i + std::size_t(j) * n]);
	    }
	    A0[j + std::size_t(j) * n] =
	        T(std::real(A0[j + std::size_t(j) * n]) + n);
	}
    }
    auto call = [&]() {
	return lu ? getrf(n, A.data(), piv.data()) : potrf(n, A.data());
    };
    for (int w = 0; w < opt.warmup; ++w) {
	A = A0;
	call();
    }
    std::vector<double> t;
    double spent = 0, res = -1;
    int info0 = 0;
    while ((int)t.size() < opt.maxSamples &&
           ((int)t.size() < opt.minSamples || spent < opt.minTime)) {
	std::memcpy(A.data(), A0.data(), nn * sizeof(T));
	const double t0 = now();
	const int info = call();
	const double t1 = now();
	t.push_back(t1 - t0);
	spent += t1 - t0;
	if (t.size() == 1) {
	    info0 = info;
	    res = info ? 1e300 : residual(lu, n, A0, A, piv);
	}
    }
    std::sort(t.begin(), t.end());
    const std::size_t k = t.size();
    const double med = (k % 2) ? t[k / 2] : 0.5 * (t[k / 2 - 1] + t[k / 2]);
    const double flops =
        (lu ? 2.0 / 3 : 1.0 / 3) * double(n) * n * n * (isComplex<T>() ? 4 : 1);
#if defined(BENCH_ULMBLAS)
    const int threads = ulmBLAS::numThreads();
#elif defined(BENCH_OPENBLAS)
    const int threads = openblas_get_num_threads();
#else
    const int threads = MKL_Get_Max_Threads();
#endif
    std::printf("%s %s %s %d %d %.6e %.6e %.6e %zu %.3f %s %.3g\n", prec,
                lu ? "getrf" : "potrf", LIBNAME, n, threads, med, t.front(),
                t.back(), k, flops / med * 1e-9,
                (info0 == 0 && res < 30) ? "ok" : "FAILED", res);
    std::fflush(stdout);
}

} // namespace

int
main(int argc, char **argv)
{
    if (argc < 6) {
	std::fprintf(stderr,
	             "usage: %s s|d|c|z getrf|potrf N0 N1 STEP "
	             "[options]\n",
	             argv[0]);
	return 2;
    }
    const char *prec = argv[1];
    const bool lu = !std::strcmp(argv[2], "getrf");
    if ((!lu && std::strcmp(argv[2], "potrf")) || std::strlen(prec) != 1 ||
        !std::strchr("sdcz", prec[0])) {
	std::fprintf(stderr, "unknown precision or routine\n");
	return 2;
    }
    const int n0 = std::atoi(argv[3]), n1 = std::atoi(argv[4]),
              step = std::atoi(argv[5]);
    Options opt;
    for (int i = 6; i + 1 < argc; i += 2) {
	const std::string a = argv[i];
	const char *v = argv[i + 1];
	if (a == "--min-time")
	    opt.minTime = std::atof(v);
	else if (a == "--min-samples")
	    opt.minSamples = std::atoi(v);
	else if (a == "--max-samples")
	    opt.maxSamples = std::atoi(v);
	else if (a == "--warmup")
	    opt.warmup = std::atoi(v);
	else {
	    std::fprintf(stderr, "unknown option %s\n", a.c_str());
	    return 2;
	}
    }
    std::printf("# prec routine lib n threads median min max samples gflops "
                "check residual\n");
    for (int n = n0; n <= n1; n += step) {
	switch (prec[0]) {
	case 's':
	    run<float>(prec, lu, n, opt);
	    break;
	case 'd':
	    run<double>(prec, lu, n, opt);
	    break;
	case 'c':
	    run<cfloat>(prec, lu, n, opt);
	    break;
	default:
	    run<cdouble>(prec, lu, n, opt);
	    break;
	}
    }
    return 0;
}
