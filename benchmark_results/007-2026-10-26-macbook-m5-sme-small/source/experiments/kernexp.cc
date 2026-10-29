//
//  kernexp: where does the time of one call of the SME2 micro kernel of
//  ulmBLAS (ugemm_32x16, master) go for small kc?  ns per call, full 32 x
//  16 tile, alpha = 1, C, A, B hot in the caches, median of 7 rounds.
//
//      lib     ulmBLAS::sme::ugemm_32x16 itself (beta = 1: C loaded into ZA,
//              stored from ZA, 48 prefetches of the next tile)
//      beta0   the same with beta = 0 (ZA zeroed instead of loaded)
//      nopf    copy without the prefetches
//      nocl    copy without the prefetches, C neither loaded nor stored
//              (only one slice stored)
//      empty   copy with kc = 0 (fixed costs only: streaming mode, ZA,
//              C load and store, prefetches)
//
//  Build: clang++ -std=c++17 -O2 -mcpu=apple-m4 -I<ulmBLAS> kernexp.cc
//
#include <ulmblas/ulmblas.h>
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <vector>

static double
now()
{
    timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

using namespace ulmBLAS::sme;

__attribute__((noinline)) __arm_new("za") __arm_locally_streaming
void
copy(long kc, const double *A, const double *B, double *C, long ldC,
     bool prefetch, bool loadC, bool storeC)
{
    const svbool_t  p  = svptrue_b64();
    const svcount_t pc = svptrue_c64();

    if (prefetch) {
        for (long c=0; c<16; ++c) {
            __builtin_prefetch(C + c*ldC + 32, 1, 2);
            __builtin_prefetch(C + c*ldC + 48, 1, 2);
            __builtin_prefetch(C + c*ldC + 63, 1, 2);
        }
    }
    svzero_za();
    if (loadC) {
        load_32x8<0>(8, C, ldC, p, p, p, p);
        load_32x8<4>(8, C + 8*ldC, ldC, p, p, p, p);
    }
    for (long l=0; l<kc; ++l) {
        svfloat64x4_t a = svld1_f64_x4(pc, A);
        svfloat64x2_t b = svld1_f64_x2(pc, B);
        svmopa_za64_f64_m(0, p, p, svget2(b, 0), svget4(a, 0));
        svmopa_za64_f64_m(1, p, p, svget2(b, 0), svget4(a, 1));
        svmopa_za64_f64_m(2, p, p, svget2(b, 0), svget4(a, 2));
        svmopa_za64_f64_m(3, p, p, svget2(b, 0), svget4(a, 3));
        svmopa_za64_f64_m(4, p, p, svget2(b, 1), svget4(a, 0));
        svmopa_za64_f64_m(5, p, p, svget2(b, 1), svget4(a, 1));
        svmopa_za64_f64_m(6, p, p, svget2(b, 1), svget4(a, 2));
        svmopa_za64_f64_m(7, p, p, svget2(b, 1), svget4(a, 3));
        A += 32;
        B += 16;
    }
    if (storeC) {
        store_32x8<0>(8, C, ldC, p, p, p, p);
        store_32x8<4>(8, C + 8*ldC, ldC, p, p, p, p);
    } else {
        svst1_hor_za64(0, 0, p, C);
    }
}

__attribute__((noinline)) __arm_new("za") __arm_locally_streaming
void
copyVec(long kc, const double *A, const double *B, double *C, long ldC)
{
    const svbool_t  p  = svptrue_b64();
    const svcount_t pc = svptrue_c64();

    for (long s=0; s<8; ++s) {
        // column s and 8+s of C (32 values each) into slice s of the tiles
        svfloat64x4_t x = svld1_f64_x4(pc, C + s*ldC);
        svfloat64x4_t y = svld1_f64_x4(pc, C + (8+s)*ldC);
        svwrite_hor_za64_f64_m(0, s, p, svget4(x, 0));
        svwrite_hor_za64_f64_m(1, s, p, svget4(x, 1));
        svwrite_hor_za64_f64_m(2, s, p, svget4(x, 2));
        svwrite_hor_za64_f64_m(3, s, p, svget4(x, 3));
        svwrite_hor_za64_f64_m(4, s, p, svget4(y, 0));
        svwrite_hor_za64_f64_m(5, s, p, svget4(y, 1));
        svwrite_hor_za64_f64_m(6, s, p, svget4(y, 2));
        svwrite_hor_za64_f64_m(7, s, p, svget4(y, 3));
    }
    for (long l=0; l<kc; ++l) {
        svfloat64x4_t a = svld1_f64_x4(pc, A);
        svfloat64x2_t b = svld1_f64_x2(pc, B);
        svmopa_za64_f64_m(0, p, p, svget2(b, 0), svget4(a, 0));
        svmopa_za64_f64_m(1, p, p, svget2(b, 0), svget4(a, 1));
        svmopa_za64_f64_m(2, p, p, svget2(b, 0), svget4(a, 2));
        svmopa_za64_f64_m(3, p, p, svget2(b, 0), svget4(a, 3));
        svmopa_za64_f64_m(4, p, p, svget2(b, 1), svget4(a, 0));
        svmopa_za64_f64_m(5, p, p, svget2(b, 1), svget4(a, 1));
        svmopa_za64_f64_m(6, p, p, svget2(b, 1), svget4(a, 2));
        svmopa_za64_f64_m(7, p, p, svget2(b, 1), svget4(a, 3));
        A += 32;
        B += 16;
    }
    for (long s=0; s<8; ++s) {
        svfloat64_t z = svdup_f64(0.0);
        svfloat64x4_t x = svcreate4(svread_hor_za64_f64_m(z, p, 0, s),
                                    svread_hor_za64_f64_m(z, p, 1, s),
                                    svread_hor_za64_f64_m(z, p, 2, s),
                                    svread_hor_za64_f64_m(z, p, 3, s));
        svfloat64x4_t y = svcreate4(svread_hor_za64_f64_m(z, p, 4, s),
                                    svread_hor_za64_f64_m(z, p, 5, s),
                                    svread_hor_za64_f64_m(z, p, 6, s),
                                    svread_hor_za64_f64_m(z, p, 7, s));
        svst1_f64_x4(pc, C + s*ldC, x);
        svst1_f64_x4(pc, C + (8+s)*ldC, y);
    }
}

//  the same prefetches, issued in ordinary mode before the kernel
__attribute__((noinline))
void
pfNormal(long kc, const double *A, const double *B, double *C, long ldC)
{
    for (long c=0; c<16; ++c) {
        __builtin_prefetch(C + c*ldC + 32, 1, 2);
        __builtin_prefetch(C + c*ldC + 48, 1, 2);
        __builtin_prefetch(C + c*ldC + 63, 1, 2);
    }
    copy(kc, A, B, C, ldC, false, true, true);
}

int
main()
{
    const long ldC = 64, KMAX = 256;
    std::vector<double> A(32*KMAX, 0.001), B(16*KMAX, 0.002), C(ldC*16, 0.0);
    const long REP = 100000;
    printf("ns per call of the 32 x 16 SME2 micro kernel (median of 7)\n");
    printf("  kc     lib   beta0    nopf    nocl  vec(nopf)  pf-ordinary  fmopa-only-est\n");
    for (long kc : {0L, 8L, 16L, 32L, 64L, 256L}) {
        std::vector<double> t[6];
        for (int round=0; round<7; ++round) {
            for (int v=0; v<6; ++v) {
                double t0 = now();
                for (long r=0; r<REP; ++r) {
                    switch (v) {
                        case 0: ugemm_32x16(kc, 1.0, A.data(), B.data(), 1.0,
                                            C.data(), ldC, 32L, 16L); break;
                        case 1: ugemm_32x16(kc, 1.0, A.data(), B.data(), 0.0,
                                            C.data(), ldC, 32L, 16L); break;
                        case 2: copy(kc, A.data(), B.data(), C.data(), ldC,
                                     false, true, true); break;
                        case 3: copy(kc, A.data(), B.data(), C.data(), ldC,
                                     false, false, false); break;
                        case 4: copyVec(kc, A.data(), B.data(), C.data(),
                                        ldC); break;
                        case 5: pfNormal(kc, A.data(), B.data(), C.data(),
                                         ldC); break;
                    }
                }
                t[v].push_back(1e9*(now()-t0)/REP);
            }
        }
        double m[6];
        for (int v=0; v<6; ++v) {
            std::sort(t[v].begin(), t[v].end());
            m[v] = t[v][3];
        }
        // FMOPA throughput measured in the campaign: 542 GFLOP/s, 128 flops
        printf("  %3ld  %6.1f  %6.1f  %6.1f  %6.1f  %9.1f  %11.1f  %6.1f\n", kc,
               m[0], m[1], m[2], m[3], m[4], m[5], kc*8*128/542.0);
    }
    return C[0]==-1;
}
