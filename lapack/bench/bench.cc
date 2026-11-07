//
//  Benchmark of dgetrf / dpotrf: ulmlapack (BENCH_ULMBLAS) or the LAPACK
//  routines of OpenBLAS (BENCH_OPENBLAS).  Same source, same timing.
//
//      bench-ulmblas  getrf|potrf N0 N1 STEP [options]
//      bench-openblas getrf|potrf N0 N1 STEP [options]
//
//  options:  --min-time S     sample until S seconds are spent (0.3)
//            --min-samples K  at least K samples (5)
//            --max-samples K  at most K samples (200)
//            --warmup K       untimed calls before sampling (2)
//            --raw FILE       append every sample to FILE
//            --nb, --ib, --pt, --serial-n, --threads   ulmlapack parameters
//            --empty 1        ulmlapack: empty tasks (cost of the task graph)
//            --inc-row R      ulmlapack: row stride R (1)
//            --variant V      ulmlapack: inplace (default), colmajor (the task
//                             graph works on a contiguous column-major copy),
//                             tiled (on a copy in tile layout; potrf only);
//                             the copies are part of the measured call
//            --prepack 1      ulmlapack dpotrf: packed tiles (param prepack)
//            --outer S        ulmlapack dpotrf in place: outer blocks of S tiles
//            --window W       with --outer: trailing updates every W blocks
//            --strip 1        with --outer: factor the block on contiguous tiles
//
//  Matrices: m = n, ldA = n, column major.  With --inc-row R > 1 the matrix
//  is stored with row stride R and column stride R*n (every R-th row of a
//  column-major matrix: A(i,j) = A[i*R + j*R*n]) and factored by the C++
//  interface (ulmlapack.hpp); R = 1 uses the C interface.  dgetrf: entries uniform in
//  [-1,1] (pivoting needed); dpotrf: symmetric, entries uniform in [-1,1],
//  n added to the diagonal (positive definite), lower triangle.
//
//  A sample is the time of ONE call of the routine (steady_clock) on a fresh
//  copy of the matrix; the copy (memcpy) is made before the clock starts.
//  Everything the routine does (task creation, parallel region, data
//  movement) is inside the measured time; the generation of the matrix is
//  not.  The result of the first timed call is checked with a solve
//  (backward error of A x = b, O(n^2)).
//
//  Output (one line per n):
//      routine lib n threads nb ib pt serial_n median min max samples
//      gflops(median) check backward_error inc_row variant
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#ifdef BENCH_ULMBLAS
#   include <ulmlapack.h>
#   include <ulmlapack.hpp>
#   ifdef ULMLAPACK_CHECK_DEPS
extern "C" void ulmlapack_last_profile(long *, int *, double *, double *,
                                       double *);
#   endif
#   define LIBNAME "ulmblas"
#elif defined(BENCH_OPENBLAS)
#   define LIBNAME "openblas"
#else
#   error "BENCH_ULMBLAS or BENCH_OPENBLAS"
#endif

extern "C" {
void dgetrf_(const int *m, const int *n, double *A, const int *ldA,
             int *ipiv, int *info);
void dpotrf_(const char *uplo, const int *n, double *A, const int *ldA,
             int *info, std::size_t len);
void dgemv_(const char *trans, const int *m, const int *n,
            const double *alpha, const double *A, const int *ldA,
            const double *x, const int *incx, const double *beta,
            double *y, const int *incy, std::size_t len);
int  openblas_get_num_threads(void);
char *openblas_get_config(void);
char *openblas_get_corename(void);
}

namespace {

double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Options
{
    double      minTime    = 0.3;
    int         minSamples = 5;
    int         maxSamples = 200;
    int         warmup     = 2;
    std::string raw;
    long        incRow     = 1;
    std::string variant    = "inplace";
#ifdef BENCH_ULMBLAS
    ulmlapack_param p;
#endif
};

//  backward error ||A x - b||_inf / (||A||_inf ||x||_inf n eps) for the
//  solution x of A x = b computed with the factors F
double
check(bool lu, int n, const double *A0, const double *F, const int *ipiv)
{
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double> x(n), b(n);
    for (int i=0; i<n; ++i) {
        b[i] = u(rng);
    }
    x = b;
    if (lu) {
        // structure: pivots in range, |L| <= 1 (partial pivoting)
        for (int j=0; j<n; ++j) {
            if (ipiv[j]<j+1 || ipiv[j]>n) {
                return 1e300;
            }
            for (int i=j+1; i<n; ++i) {
                if (!(std::fabs(F[i+std::size_t(j)*n])<=1)) {
                    return 1e300;
                }
            }
        }
        for (int i=0; i<n; ++i) {
            std::swap(x[i], x[ipiv[i]-1]);
        }
        for (int j=0; j<n; ++j) {                   // L y = P b
            const double xj = x[j];
            for (int i=j+1; i<n; ++i) {
                x[i] -= F[i+std::size_t(j)*n]*xj;
            }
        }
        for (int j=n-1; j>=0; --j) {                // U x = y
            x[j] /= F[j+std::size_t(j)*n];
            const double xj = x[j];
            for (int i=0; i<j; ++i) {
                x[i] -= F[i+std::size_t(j)*n]*xj;
            }
        }
    } else {
        for (int j=0; j<n; ++j) {                   // L y = b
            x[j] /= F[j+std::size_t(j)*n];
            const double xj = x[j];
            for (int i=j+1; i<n; ++i) {
                x[i] -= F[i+std::size_t(j)*n]*xj;
            }
        }
        for (int i=n-1; i>=0; --i) {                // L^T x = y
            double s = x[i];
            for (int l=i+1; l<n; ++l) {
                s -= F[l+std::size_t(i)*n]*x[l];
            }
            x[i] = s/F[i+std::size_t(i)*n];
        }
    }
    // r = A x - b (A0 full: for potrf symmetric, both triangles were set)
    std::vector<double> r(b);
    const double one = 1, mone = -1;
    const int inc = 1;
    dgemv_("N", &n, &n, &one, A0, &n, x.data(), &inc, &mone, r.data(), &inc, 1);
    double nr = 0, nx = 0, na = 0;
    for (int i=0; i<n; ++i) {
        nr = std::max(nr, std::fabs(r[i]));
        nx = std::max(nx, std::fabs(x[i]));
        double s = 0;
        for (int j=0; j<n; ++j) {
            s += std::fabs(A0[i+std::size_t(j)*n]);
        }
        na = std::max(na, s);
    }
    if (!std::isfinite(nx) || !std::isfinite(nr)) {
        return 1e300;
    }
    return nr/(na*nx*n*2.220446049250313e-16);
}

int
factor(bool lu, int n, double *A, int *ipiv, const Options &opt)
{
#ifdef BENCH_ULMBLAS
    if (opt.incRow!=1 || opt.variant!="inplace") {
        static std::vector<ulmlapack::Index> piv;
        piv.resize(n);
        const long ir = opt.incRow, ic = opt.incRow*n;
        if (!lu) {
            return (int) (opt.variant=="colmajor"
                          ? ulmlapack::potrf_colmajor(true, n, A, ir, ic, &opt.p)
                          : opt.variant=="tiled"
                          ? ulmlapack::potrf_tiled(true, n, A, ir, ic, &opt.p)
                          : ulmlapack::potrf(true, n, A, ir, ic, &opt.p));
        }
        const int info = (int) (opt.variant=="colmajor"
                          ? ulmlapack::getrf_colmajor(n, n, A, ir, ic,
                                                      piv.data(), 1, &opt.p)
                          : ulmlapack::getrf(n, n, A, ir, ic, piv.data(), 1,
                                             &opt.p));
        for (int i=0; i<n; ++i) {
            ipiv[i] = int(piv[i]+1);
        }
        return info;
    }
    return lu ? ulmlapack_dgetrf_param(n, n, A, n, ipiv, &opt.p)
              : ulmlapack_dpotrf_param('L', n, A, n, &opt.p);
#else
    (void) opt;
    int info;
    if (lu) {
        dgetrf_(&n, &n, A, &n, ipiv, &info);
    } else {
        dpotrf_("L", &n, A, &n, &info, 1);
    }
    return info;
#endif
}

void
run(bool lu, int n, const Options &opt, FILE *raw)
{
    std::mt19937_64 rng(4711+n);
    std::uniform_real_distribution<double> u(-1, 1);
    const std::size_t nn = std::size_t(n)*n;
    const long        R  = opt.incRow;
    std::vector<double> A0(nn), A(R==1 ? nn : 0);
    std::vector<int>    ipiv(n);
    //  strided storage (R > 1): A(i,j) = W[i*R + j*R*n]; not initialized,
    //  the gaps are never touched
    const std::size_t extent = (R==1) ? 0 : std::size_t(R)*nn;
    std::unique_ptr<double[]> W(extent ? new double[extent] : 0);
    double *work = (R==1) ? A.data() : W.get();
    auto fresh = [&]() {
        if (R==1) {
            std::memcpy(A.data(), A0.data(), nn*sizeof(double));
        } else {
            for (int j=0; j<n; ++j) {
                double       *w = &W[std::size_t(j)*R*n];
                const double *a = &A0[std::size_t(j)*n];
                for (int i=0; i<n; ++i) {
                    w[std::size_t(i)*R] = a[i];
                }
            }
        }
    };
    for (std::size_t i=0; i<nn; ++i) {
        A0[i] = u(rng);
    }
    if (!lu) {
        for (int j=0; j<n; ++j) {
            for (int i=j+1; i<n; ++i) {
                A0[j+std::size_t(i)*n] = A0[i+std::size_t(j)*n];
            }
            A0[j+std::size_t(j)*n] += n;
        }
    }
    for (int w=0; w<opt.warmup; ++w) {
        fresh();
        factor(lu, n, work, ipiv.data(), opt);
    }
    std::vector<double> t;
    double spent = 0, chk = -1;
    int info0 = 0;
    while ((int) t.size()<opt.maxSamples
           && ((int) t.size()<opt.minSamples || spent<opt.minTime))
    {
        fresh();
        const double t0 = now();
        const int info = factor(lu, n, work, ipiv.data(), opt);
        const double t1 = now();
        t.push_back(t1-t0);
        spent += t1-t0;
        if (t.size()==1) {
            info0 = info;
            std::vector<double> F(nn);
            for (int j=0; j<n; ++j) {
                for (int i=0; i<n; ++i) {
                    F[i+std::size_t(j)*n] = work[std::size_t(i)*R
                                                 +std::size_t(j)*R*n];
                }
            }
            chk = info ? 1e300 : check(lu, n, A0.data(), F.data(), ipiv.data());
        }
    }
    if (raw) {
        for (std::size_t s=0; s<t.size(); ++s) {
            std::fprintf(raw, "%s %s %d %zu %.9e %ld %s\n", lu ? "getrf" : "potrf",
                         LIBNAME, n, s, t[s], R, opt.variant.c_str());
        }
    }
    std::vector<double> ts(t);
    std::sort(ts.begin(), ts.end());
    const std::size_t k = ts.size();
    const double med = (k%2) ? ts[k/2] : 0.5*(ts[k/2-1]+ts[k/2]);
    const double flops = (lu ? 2.0/3 : 1.0/3)*double(n)*n*n;

    int threads, nb = 0, ib = 0, pt = 0, sn = 0;
#ifdef BENCH_ULMBLAS
    ulmlapack_param p = opt.p;
    if (lu) {
        ulmlapack_dgetrf_resolve(n, n, &p);
    } else {
        ulmlapack_dpotrf_resolve(n, &p);
    }
    threads = p.threads;
    nb = p.nb; ib = p.ib; pt = p.panel_threads; sn = p.serial_n;
    if (n<=sn || p.threads==1 || n<=p.nb) {
        threads = 1;                // serial algorithm (see ulmlapack)
    }
#else
    threads = openblas_get_num_threads();
#endif
    std::printf("%s %s %d %d %d %d %d %d %.6e %.6e %.6e %zu %.3f %s",
                lu ? "getrf" : "potrf", LIBNAME, n, threads, nb, ib, pt, sn,
                med, ts.front(), ts.back(), k, flops/med*1e-9,
                (info0==0 && chk<100) ? "ok" : "FAILED");
#ifdef BENCH_ULMBLAS
    const bool pp = opt.p.prepack!=0;
    const int  so = opt.p.outer;
#else
    const bool pp = false;
    const int  so = 0;
#endif
    std::printf(" %.3g %ld %s%s", chk, R, opt.variant.c_str(), pp ? "+pp" : "");
    if (so>1) {
#ifdef BENCH_ULMBLAS
        std::printf("+outer%d", so);
        if (opt.p.window>1) {
            std::printf("+w%d", opt.p.window);
        }
        if (opt.p.strip) {
            std::printf("+strip");
        }
#endif
    }
    std::printf("\n");
#if defined(BENCH_ULMBLAS) && defined(ULMLAPACK_CHECK_DEPS)
    {   // profile of the last call (diagnostic build)
        long tasks; int th; double gen, wall, busy;
        ulmlapack_last_profile(&tasks, &th, &gen, &wall, &busy);
        std::printf("# profile n=%d tasks %ld threads %d generate %.6e "
                    "wall %.6e busy %.6e\n", n, tasks, th, gen, wall, busy);
    }
#endif
    std::fflush(stdout);
}

} // namespace

int
main(int argc, char **argv)
{
    if (argc<5) {
        std::fprintf(stderr, "usage: %s getrf|potrf N0 N1 STEP [options]\n",
                     argv[0]);
        return 2;
    }
    const bool lu = !std::strcmp(argv[1], "getrf");
    if (!lu && std::strcmp(argv[1], "potrf")) {
        std::fprintf(stderr, "unknown routine %s\n", argv[1]);
        return 2;
    }
    const int n0 = std::atoi(argv[2]), n1 = std::atoi(argv[3]),
              step = std::atoi(argv[4]);
    Options opt;
#ifdef BENCH_ULMBLAS
    ulmlapack_param_init(&opt.p);
#endif
    for (int i=5; i<argc; ++i) {
        const std::string a = argv[i];
        if (i+1>=argc) {
            std::fprintf(stderr, "missing value for %s\n", a.c_str());
            return 2;
        }
        const char *v = argv[++i];
        if (a=="--min-time")         opt.minTime    = std::atof(v);
        else if (a=="--min-samples") opt.minSamples = std::atoi(v);
        else if (a=="--max-samples") opt.maxSamples = std::atoi(v);
        else if (a=="--warmup")      opt.warmup     = std::atoi(v);
        else if (a=="--raw")         opt.raw        = v;
        else if (a=="--inc-row")     opt.incRow     = std::atol(v);
        else if (a=="--variant")     opt.variant    = v;
#ifdef BENCH_ULMBLAS
        else if (a=="--nb")          opt.p.nb            = std::atoi(v);
        else if (a=="--ib")          opt.p.ib            = std::atoi(v);
        else if (a=="--pt")          opt.p.panel_threads = std::atoi(v);
        else if (a=="--serial-n")    opt.p.serial_n      = std::atoi(v);
        else if (a=="--threads")     opt.p.threads       = std::atoi(v);
        else if (a=="--empty")       opt.p.empty_tasks   = std::atoi(v);
        else if (a=="--prepack")     opt.p.prepack       = std::atoi(v);
        else if (a=="--outer")       opt.p.outer         = std::atoi(v);
        else if (a=="--window")      opt.p.window        = std::atoi(v);
        else if (a=="--strip")       opt.p.strip         = std::atoi(v);
#endif
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (opt.incRow<1) {
        std::fprintf(stderr, "--inc-row must be >= 1\n");
        return 2;
    }
    if (opt.variant!="inplace" && opt.variant!="colmajor"
        && !(opt.variant=="tiled" && !lu))
    {
        std::fprintf(stderr, "unknown variant %s (tiled: potrf only)\n",
                     opt.variant.c_str());
        return 2;
    }
#ifdef BENCH_OPENBLAS
    if (opt.incRow!=1 || opt.variant!="inplace") {
        std::fprintf(stderr, "OpenBLAS's LAPACK: --inc-row 1, inplace only\n");
        return 2;
    }
    std::printf("# openblas: %s, core %s, threads %d\n", openblas_get_config(),
                openblas_get_corename(), openblas_get_num_threads());
#else
    std::printf("# ulmlapack, omp_get_max_threads %d, max task priority %d\n",
                omp_get_max_threads(), omp_get_max_task_priority());
#endif
    std::printf("# routine lib n threads nb ib pt serial_n median min max "
                "samples gflops check backward_error inc_row variant\n");
    FILE *raw = opt.raw.empty() ? 0 : std::fopen(opt.raw.c_str(), "a");
    for (int n=n0; n<=n1; n+=step) {
        run(lu, n, opt, raw);
    }
    if (raw) {
        std::fclose(raw);
    }
    return 0;
}
