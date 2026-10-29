/*
 *  bench-eigen: Eigen's own single-threaded matrix product, timed exactly
 *  as benchx.c times cblas_?gemm (same sizes, samples, warm-up, check and
 *  output), for one data type selected with -DPREC=s, d, c or z.  Used by
 *  run.sh (make bench-gemm).
 *
 *      bench-eigen SAMPLES N1 N2 ...   (square sizes m = n = k = Ni)
 *      bench-eigen -i                  (identity only)
 *
 *  Eigen has no CBLAS interface: the timed operation is the native C++
 *  expression C.noalias() += A*B with column-major
 *  Eigen::Matrix<T, Dynamic, Dynamic> (the GEMM C <- A*B + C, alpha = beta
 *  = 1, of benchx.c).  Eigen must not forward the product to a BLAS
 *  library or use OpenMP (checked at compile time).
 */
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#if defined(__APPLE__)
#   include <pthread/qos.h>
#endif

#if defined(EIGEN_USE_BLAS) || defined(EIGEN_USE_MKL_ALL) || defined(_OPENMP)
#error "this benchmark measures Eigen's own single-threaded product"
#endif

#define s_ 1
#define d_ 2
#define c_ 3
#define z_ 4
#define CAT_(a, b) a##b
#define CAT(a, b)  CAT_(a, b)
#define P_ CAT(PREC, _)

#if P_==1
typedef float                T;
typedef float                R;
#elif P_==2
typedef double               T;
typedef double               R;
#elif P_==3
typedef std::complex<float>  T;
typedef float                R;
#else
typedef std::complex<double> T;
typedef double               R;
#endif

static const bool isComplex = P_>=3;
typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> M;

static const int    MAX_SAMPLES = 64;
static const double MIN_SAMPLE  = 0.2;
static const int    CHECKS      = 400;

static double
now()
{
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

static R
uniform()
{
    return R(2.0*rand()/RAND_MAX - 1.0);
}

//  the same values in the same order as fill() of benchx.c
static void
fill(M &X)
{
    R *x = reinterpret_cast<R *>(X.data());
    for (long i=0; i<X.size()*(isComplex ? 2 : 1); ++i) {
        x[i] = uniform();
    }
}

static void
product(const M &A, const M &B, M &C)
{
    C.noalias() += A*B;
    asm volatile("" : : "r"(C.data()) : "memory");
}

static long double re(float x)  { return x; }
static long double re(double x) { return x; }
template <typename U> static long double re(const std::complex<U> &x) { return x.real(); }
static long double im(float)  { return 0; }
static long double im(double) { return 0; }
template <typename U> static long double im(const std::complex<U> &x) { return x.imag(); }

//  as check() of benchx.c
static double
check(int n, const M &A, const M &B, const M &C0, const M &C)
{
    const double eps = std::numeric_limits<R>::epsilon();
    double worst = 0;
    for (int c=0; c<CHECKS; ++c) {
        int i = rand() % n, j = rand() % n;
        long double sr = re(C0(i,j)), br = std::fabs(re(C0(i,j)));
        long double si = im(C0(i,j)), bi = std::fabs(im(C0(i,j)));
        for (int l=0; l<n; ++l) {
            long double ar = re(A(i,l)), ai = im(A(i,l));
            long double xr = re(B(l,j)), xi = im(B(l,j));
            sr += ar*xr - ai*xi;   br += std::fabs(ar*xr) + std::fabs(ai*xi);
            si += ar*xi + ai*xr;   bi += std::fabs(ar*xi) + std::fabs(ai*xr);
        }
        double er = std::fabs(re(C(i,j))-sr)/(4*(n+2)*eps*br+1e-300);
        double ei = isComplex ? std::fabs(im(C(i,j))-si)/(4*(n+2)*eps*bi+1e-300)
                              : 0;
        if (!std::isfinite(er) || !std::isfinite(ei)) return -1;
        worst = std::max(worst, std::max(er, ei));
    }
    return worst;
}

static void
run(int samples, int n)
{
    M      A(n, n), B(n, n), C(n, n), C0(n, n);
    double t[MAX_SAMPLES], flops = (isComplex ? 8.0 : 2.0)*n*n*n;
    long   calls = 1;

    fill(A);
    fill(B);
    fill(C0);
    C = C0;

    double t0 = now();
    product(A, B, C);
    double t1 = now();
    double err = check(n, A, B, C0, C);
    double tc = now() - t1;
    while (t1-t0 < 0.3) {
        product(A, B, C);
        ++calls;
        t1 = now() - tc;
    }
    calls = long(calls*MIN_SAMPLE/(t1-t0)) + 1;

    for (int s=0; s<samples; ++s) {
        double t0 = now();
        for (long c=0; c<calls; ++c) {
            product(A, B, C);
        }
        t[s] = flops*calls/(now()-t0)*1e-9;
    }
    std::sort(t, t+samples);
    std::printf("%5d %5d %5d %8.3f %8.3f %8.3f %6ld %8.4f %s\n", n, n, n,
                t[samples/2], t[0], t[samples-1], calls, err,
                (err>=0 && err<=1) ? "ok" : "WRONG");
    std::fflush(stdout);
}

int
main(int argc, char **argv)
{
#if defined(__APPLE__)
    //  macOS: run on a performance core
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    std::printf("# routine Eigen C.noalias() += A*B, %s, %s flops\n",
                P_==1 ? "float" : P_==2 ? "double"
                      : P_==3 ? "complex float" : "complex double",
                isComplex ? "8mnk" : "2mnk");
    //  Eigen 5 keeps the legacy macros (3.5.0) and adds EIGEN_VERSION_STRING
#if defined(EIGEN_VERSION_STRING)
    std::printf("# library Eigen %s (header only, SIMD: %s)\n",
                EIGEN_VERSION_STRING, Eigen::SimdInstructionSetsInUse());
#else
    std::printf("# library Eigen %d.%d.%d (header only, SIMD: %s)\n",
                EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION,
                Eigen::SimdInstructionSetsInUse());
#endif
    std::fflush(stdout);
    if (argc==2 && !std::strcmp(argv[1], "-i")) {
        return 0;
    }
    if (argc<3) {
        std::fprintf(stderr, "usage: %s SAMPLES N1 N2 ...\n", argv[0]);
        return 2;
    }
    int samples = std::atoi(argv[1]);
    if (samples<1 || samples>MAX_SAMPLES) {
        return 2;
    }
    srand(2026);
    for (int i=2; i<argc; ++i) {
        run(samples, std::atoi(argv[i]));
    }
    return 0;
}
