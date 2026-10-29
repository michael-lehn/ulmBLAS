/*
 *  bench-eigen: single-threaded Eigen matrix product, timed exactly like
 *  bench2.c (campaign 001-2026-10-24-heim-avx-fma, run B) times cblas_dgemm.
 *
 *      bench-eigen SAMPLES BETA M N K [M N K ...]
 *
 *  A (m x k), B (k x n), C (m x n) are
 *  Eigen::Matrix<double, Dynamic, Dynamic, ColMajor>, i.e. column major with
 *  leading dimension m, k, m as in bench2.  The timed operation is
 *
 *      BETA = 1:  C.noalias() += A*B;     (dgemm, alpha = 1, beta = 1)
 *      BETA = 0:  C.noalias()  = A*B;     (dgemm, alpha = 1, beta = 0)
 *
 *  A, B and C0 receive the same values as in bench2 (same calls of rand()
 *  in the same order), and the result of the first call is checked with
 *  the same naive product and error bound.  The check also fails if the
 *  call changed A or B or moved the storage of C.  Warm-up of at least
 *  0.3 s, then SAMPLES samples of at least MIN_SAMPLE seconds each; with
 *  BETA = 0, C is filled with NaN before each sample.  All matrices are
 *  allocated before the timed loop; the timed loop contains only the
 *  product.  The empty asm statement after each product makes the
 *  compiler assume that C may be read and changed (as by a call into a
 *  library), so that products of inlined small sizes are neither merged
 *  nor removed.  Output per size (as bench2):
 *
 *      m n k beta median_GFLOPS min max calls_per_sample check
 */
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#if defined(EIGEN_USE_BLAS) || defined(EIGEN_USE_MKL_ALL) || defined(_OPENMP)
#error "this benchmark measures Eigen's own single-threaded product"
#endif

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor>;

#define MAX_SAMPLES 64
#define MIN_SAMPLE  0.2

static double
now()
{
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

static void
fill(Matrix &X, int nanValues)
{
    double *x = X.data();
    for (long i=0; i<X.size(); ++i) {
        x[i] = nanValues ? NAN : (double)rand()/RAND_MAX - 0.5;
    }
}

static void
clobber(Matrix &C)
{
    asm volatile("" : : "r"(C.data()) : "memory");
}

static void
product(double beta, const Matrix &A, const Matrix &B, Matrix &C)
{
    if (beta==0) {
        C.noalias() = A*B;
    } else {
        C.noalias() += A*B;
    }
    clobber(C);
}

static const char *
check(int m, int n, int k, const Matrix &A_, const Matrix &B_, double beta,
      const Matrix &C0_, const Matrix &C_)
{
    if ((double)m*n*k > 2e8) {
        return "unchecked";
    }
    const double *A = A_.data(), *B = B_.data(), *C0 = C0_.data(),
                 *C = C_.data();
    for (int j=0; j<n; ++j) {
        for (int i=0; i<m; ++i) {
            double s = 0, bound = 0;
            for (int l=0; l<k; ++l) {
                s     += A[i+(long)l*m]*B[l+(long)j*k];
                bound += fabs(A[i+(long)l*m]*B[l+(long)j*k]);
            }
            if (beta!=0) {
                s     += beta*C0[i+(long)j*m];
                bound += fabs(beta*C0[i+(long)j*m]);
            }
            if (!(fabs(C[i+(long)j*m]-s) <= 4*(k+2)*2.3e-16*bound+1e-300)) {
                return "WRONG";
            }
        }
    }
    return "ok";
}

static void
run(int samples, double beta, int m, int n, int k)
{
    Matrix A(m, k), B(k, n), C(m, n), C0(m, n);
    double t[MAX_SAMPLES], flops = 2.0*m*n*k;
    long   calls = 0;

    fill(A, 0);
    fill(B, 0);
    fill(C0, beta==0);
    C = C0;
    const Matrix  A0 = A, B0 = B;
    const double *Cdata = C.data();

    product(beta, A, B, C);
    const char *ok = check(m, n, k, A, B, beta, C0, C);
    if (C.data()!=Cdata || A!=A0 || B!=B0) {
        ok = "WRONG";
    }

    double t0 = now(), t1;
    do {
        product(beta, A, B, C);
        ++calls;
        t1 = now();
    } while (t1-t0 < 0.3);
    calls = (long)(calls*MIN_SAMPLE/(t1-t0)) + 1;

    for (int s=0; s<samples; ++s) {
        if (beta==0) {
            fill(C, 1);
        }
        t0 = now();
        for (long c=0; c<calls; ++c) {
            product(beta, A, B, C);
        }
        t[s] = flops*calls/(now()-t0)*1e-9;
    }
    std::sort(t, t+samples);
    printf("%5d %5d %5d %g %8.3f %8.3f %8.3f %6ld %s\n", m, n, k, beta,
           t[samples/2], t[0], t[samples-1], calls, ok);
    fflush(stdout);
}

int
main(int argc, char **argv)
{
    if (argc<6 || (argc-3)%3) {
        fprintf(stderr, "usage: %s SAMPLES BETA M N K [M N K ...]\n", argv[0]);
        return 2;
    }
    int    samples = atoi(argv[1]);
    double beta    = atof(argv[2]);
    if (samples<1 || samples>MAX_SAMPLES || (beta!=0 && beta!=1)) {
        return 2;
    }
    Eigen::setNbThreads(1);
    for (int i=3; i+2<argc; i+=3) {
        run(samples, beta, atoi(argv[i]), atoi(argv[i+1]), atoi(argv[i+2]));
    }
    return 0;
}
