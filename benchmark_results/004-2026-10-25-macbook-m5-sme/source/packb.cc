//
//  packb: throughput of packing a block of B (kc x nc, column major with
//  leading dimension ldb) into panels of NR = 16 columns.
//
//      packb KC NC LDB
//
//  Variants:
//    ulm      ulmBLAS::gepack_B (the library's routine)
//    neon     per panel: for two rows k, k+1 one 128-bit load from each of
//             the 16 columns, 2 x 2 transposes (zip1, zip2), two rows of
//             the panel stored
//    neonpfD  as neon, plus PRFM D*8 rows ahead in each column
//  B has nc*NB columns (NB blocks are packed in turn).  Output: variant
//  GB/s (bytes read + written).
//
#include <arm_neon.h>
#include <arm_sme.h>
#include <pthread/qos.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ulmblas/level3/level3.h>

static double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static void
neon(long kc, long nc, const double *B, long ldb, double *p, long D)
{
    for (long j=0; j<nc; j+=16) {
        const double *b = B + j*ldb;
        long          k = 0;
        for (; k+2<=kc; k+=2) {
            if (D && (k%8)==0) {
                for (long c=0; c<16; ++c) {
                    __builtin_prefetch(b + c*ldb + k + 8*D, 0, 3);
                }
            }
            for (long c=0; c<16; c+=2) {
                float64x2_t x = vld1q_f64(b + c*ldb + k);
                float64x2_t y = vld1q_f64(b + (c+1)*ldb + k);
                vst1q_f64(p + c, vzip1q_f64(x, y));
                vst1q_f64(p + 16 + c, vzip2q_f64(x, y));
            }
            p += 32;
        }
        for (; k<kc; ++k) {
            for (long c=0; c<16; ++c) {
                p[c] = b[c*ldb + k];
            }
            p += 16;
        }
    }
}

//  ZA as transpose buffer: per 8 rows (k) of the panel, the 8 values of
//  column c are loaded into the horizontal slice c of tile 0 (c < 8) or 1,
//  the vertical slice r is row k+r of the panel (8 + 8 values).
__arm_new("za") __arm_locally_streaming static void
smeza(long kc, long nc, const double *B, long ldb, double *p, long D)
{
    svbool_t pt = svptrue_b64();
    for (long j=0; j<nc; j+=16) {
        const double *b = B + j*ldb;
        long          k = 0;
        for (; k+8<=kc; k+=8) {
            for (unsigned c=0; c<8; ++c) {
                if (D) {
                    __builtin_prefetch(b + c*ldb + k + 8*D, 0, 3);
                    __builtin_prefetch(b + (c+8)*ldb + k + 8*D, 0, 3);
                }
                svld1_hor_za64(0, c, pt, b + c*ldb + k);
                svld1_hor_za64(1, c, pt, b + (c+8)*ldb + k);
            }
            for (unsigned r=0; r<8; ++r) {
                svst1_ver_za64(0, r, pt, p + 16*r);
                svst1_ver_za64(1, r, pt, p + 16*r + 8);
            }
            p += 128;
        }
        for (; k<kc; ++k) {
            for (long c=0; c<16; ++c) {
                p[c] = b[c*ldb + k];
            }
            p += 16;
        }
    }
}

int
main(int argc, char **argv)
{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    long kc = std::atol(argv[1]), nc = std::atol(argv[2]),
         ldb = std::atol(argv[3]);
    const long NB = 4;
    std::vector<double> B(ldb*nc*NB), P(kc*nc), Q(kc*nc);
    for (std::size_t i=0; i<B.size(); ++i) B[i] = double(i);

    auto run = [&](const char *name, int variant, long D) {
        std::vector<double> t;
        for (int s=0; s<7; ++s) {
            double t0 = now();
            for (long b=0; b<NB; ++b) {
                const double *x = &B[b*nc*ldb];
                if (variant==0) {
                    ulmBLAS::gepack_B(int(kc), int(nc), false, x, 1, int(ldb),
                                      &P[0]);
                } else if (variant==2) {
                    smeza(kc, nc, x, ldb, &P[0], D);
                } else {
                    neon(kc, nc, x, ldb, &P[0], D);
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
                    2.0*8*kc*nc*NB/t[3]*1e-9);
    };
    run("ulm", 0, 0);
    run("neon", 1, 0);
    for (long D : {1, 2, 4, 8}) {
        char n[32]; std::snprintf(n, sizeof n, "neonpf%ld", D);
        run(n, 1, D);
    }
    for (long D : {0, 1, 2, 4}) {
        char n[32]; std::snprintf(n, sizeof n, "smezapf%ld", D);
        run(n, 2, D);
    }
    run("ulm(again)", 0, 0);
    return 0;
}
