//
//  packa: throughput of packing a block of A (mc x kc, column major with
//  leading dimension lda) into panels of MR = 32 rows.
//
//      packa MC KC LDA
//
//  Variants:
//    ulm      ulmBLAS::gepack_A (the library's routine)
//    col      per panel and column a copy of 32 doubles (panel by panel)
//    colpfD   as col, plus PRFM of the same rows D columns ahead
//    grppfD   all panels of a column (as gepack_A with P = mc/MR), plus PRFM
//             of the column D columns ahead
//  The matrix A has kc*NB columns (NB blocks are packed in turn, so that A
//  is not in the cache).  Output: variant GB/s (bytes read + written).
//
#include <arm_sme.h>
#include <pthread/qos.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <ulmblas/level3/level3.h>

static double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static void
col(long mc, long kc, const double *A, long lda, double *p, long D)
{
    for (long i=0; i<mc; i+=32) {
        const double *a = A + i;
        for (long j=0; j<kc; ++j) {
            if (D) {
                __builtin_prefetch(a + D*lda, 0, 3);
                __builtin_prefetch(a + D*lda + 16, 0, 3);
                __builtin_prefetch(a + D*lda + 31, 0, 3);
            }
            std::memcpy(p, a, 32*sizeof(double));
            p += 32;
            a += lda;
        }
    }
}

__arm_locally_streaming static void
smecol(long mc, long kc, const double *A, long lda, double *p, long D)
{
    svcount_t pc = svptrue_c64();
    for (long i=0; i<mc; i+=32) {
        const double *a = A + i;
        for (long j=0; j<kc; ++j) {
            if (D) {
                __builtin_prefetch(a + D*lda, 0, 3);
                __builtin_prefetch(a + D*lda + 16, 0, 3);
                __builtin_prefetch(a + D*lda + 31, 0, 3);
            }
            svst1_f64_x4(pc, p, svld1_f64_x4(pc, a));
            p += 32;
            a += lda;
        }
    }
}

static void
grp(long mc, long kc, const double *A, long lda, double *p, long D)
{
    for (long j=0; j<kc; ++j) {
        const double *a = A + j*lda;
        if (D) {
            for (long i=0; i<mc; i+=16) {
                __builtin_prefetch(a + D*lda + i, 0, 3);
            }
        }
        for (long i=0; i<mc; i+=32) {
            std::memcpy(p + i*kc + j*32, a + i, 32*sizeof(double));
        }
    }
}

int
main(int argc, char **argv)
{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    long mc = std::atol(argv[1]), kc = std::atol(argv[2]),
         lda = std::atol(argv[3]);
    const long NB = 8;
    std::vector<double> A(lda*kc*NB), P(mc*kc), Q(mc*kc);
    for (std::size_t i=0; i<A.size(); ++i) A[i] = double(i);

    auto run = [&](const char *name, int variant, long D) {
        std::vector<double> t;
        for (int s=0; s<7; ++s) {
            double t0 = now();
            for (long b=0; b<NB; ++b) {
                const double *a = &A[b*kc*lda];
                if (variant==0) {
                    ulmBLAS::gepack_A(int(mc), int(kc), false, a, 1, int(lda),
                                      &P[0]);
                } else if (variant==1) {
                    col(mc, kc, a, lda, &P[0], D);
                } else if (variant==3) {
                    smecol(mc, kc, a, lda, &P[0], D);
                } else {
                    grp(mc, kc, a, lda, &P[0], D);
                }
            }
            t.push_back(now()-t0);
        }
        std::sort(t.begin(), t.end());
        if (variant==0) {
            Q = P;
        } else if (Q!=P) {
            std::printf("%-10s WRONG\n", name);
            return;
        }
        std::printf("%-10s %7.1f GB/s\n", name,
                    2.0*8*mc*kc*NB/t[3]*1e-9);
    };
    run("ulm", 0, 0);
    run("col", 1, 0);
    for (long D : {1, 2, 4, 8, 16}) {
        char n[32]; std::snprintf(n, sizeof n, "colpf%ld", D);
        run(n, 1, D);
    }
    for (long D : {0, 4, 8, 16}) {
        char n[32]; std::snprintf(n, sizeof n, "smecolpf%ld", D);
        run(n, 3, D);
    }
    run("grp", 2, 0);
    for (long D : {1, 2, 4, 8}) {
        char n[32]; std::snprintf(n, sizeof n, "grppf%ld", D);
        run(n, 2, D);
    }
    run("ulm(again)", 0, 0);
    return 0;
}
