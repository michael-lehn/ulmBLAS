//
//  packtail-exp: does the scalar k tail of the SME2 packing routines of
//  ulmBLAS (master 881d321, ulmblas/level3/ukernel/sme/pack.tcc) make the
//  following SME loads slow, and does a predicated vector tail avoid it?
//
//  old_pack_B_colmajor, old_pack_A_rowmajor: verbatim copies of the current
//  routines (k tail: scalar loop, compiled to integer stores of the zero
//  padding and scalar FP stores of the data).
//  new_pack_B_colmajor, new_pack_A_rowmajor: the k tail as one more block
//  through ZA, with the k range as predicate of the loads (LD1D into a ZA
//  slice sets inactive elements to zero) and only the k-l vertical slices
//  stored.
//
//  1. layout: old and new produce bit-identical buffers for nr (mr) =
//     1 ... 16 (32) and k = 1 ... 40 (incl. rows beyond the panel that must
//     stay untouched).
//  2. timing: pack a panel, then read it in a second streaming session like
//     ugemm_32x16 (svld1_f64_x2 of the 16 values of each row, FMOPA), same
//     B, same buffer, same alignment; old and new alternate, 5 rounds,
//     median.
//  3. mechanism: the full-block part packed by the new routine, then only
//     the tail rows rewritten (same addresses) with (a) integer stores
//     (stp xzr / str x), (b) scalar FP stores (str d), (c) a predicated
//     vector store (st1d), then read as in 2.
//
//  Build: clang++ -std=c++17 -O3 -mcpu=apple-m4 packtail-exp.cc -o packtail-exp
//
#include <arm_sme.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static double
now()
{
    timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

//----------------------------------------------------------------------------
//  Current code (verbatim from pack.tcc, master 881d321)
//----------------------------------------------------------------------------

__attribute__((noinline)) __arm_new("za") __arm_locally_streaming
void
old_pack_B_colmajor(long nr, long k, const double *B, long incColB,
                    double *buffer)
{
    const svbool_t p = svptrue_b64();
    long           l = 0;

    svzero_za();
    for (; l+8<=k; l+=8) {
        for (long c=0; c<8; ++c) {
            if (c  <nr) svld1_hor_za64(0, c, p, B + (c  )*incColB + l);
            if (c+8<nr) svld1_hor_za64(1, c, p, B + (c+8)*incColB + l);
        }
        for (long r=0; r<8; ++r) {
            svst1_ver_za64(0, r, p, buffer + 16*r);
            svst1_ver_za64(1, r, p, buffer + 16*r + 8);
        }
        buffer += 8*16;
    }
    for (; l<k; ++l) {
        for (long j=0; j<16; ++j) {
            buffer[j] = (j<nr) ? B[j*incColB + l] : 0.0;
        }
        buffer += 16;
    }
}

__attribute__((noinline)) __arm_new("za") __arm_locally_streaming
void
old_pack_A_rowmajor(long mr, long k, const double *A, long incRowA,
                    double *buffer)
{
    const svbool_t p = svptrue_b64();
    long           j = 0;

    svzero_za();
    for (; j+8<=k; j+=8) {
        for (long c=0; c<8; ++c) {
            if (c   <mr) svld1_hor_za64(0, c, p, A + (c   )*incRowA + j);
            if (c+ 8<mr) svld1_hor_za64(1, c, p, A + (c+ 8)*incRowA + j);
            if (c+16<mr) svld1_hor_za64(2, c, p, A + (c+16)*incRowA + j);
            if (c+24<mr) svld1_hor_za64(3, c, p, A + (c+24)*incRowA + j);
        }
        for (long r=0; r<8; ++r) {
            svst1_ver_za64(0, r, p, buffer + 32*r);
            svst1_ver_za64(1, r, p, buffer + 32*r + 8);
            svst1_ver_za64(2, r, p, buffer + 32*r + 16);
            svst1_ver_za64(3, r, p, buffer + 32*r + 24);
        }
        buffer += 8*32;
    }
    for (; j<k; ++j) {
        for (long i=0; i<32; ++i) {
            buffer[i] = (i<mr) ? A[i*incRowA + j] : 0.0;
        }
        buffer += 32;
    }
}

//----------------------------------------------------------------------------
//  Candidate fix: the k tail as one more (predicated) block through ZA
//----------------------------------------------------------------------------

__attribute__((noinline)) __arm_new("za") __arm_locally_streaming
void
new_pack_B_colmajor(long nr, long k, const double *B, long incColB,
                    double *buffer)
{
    const svbool_t p = svptrue_b64();
    long           l = 0;

    svzero_za();
    for (; l+8<=k; l+=8) {
        for (long c=0; c<8; ++c) {
            if (c  <nr) svld1_hor_za64(0, c, p, B + (c  )*incColB + l);
            if (c+8<nr) svld1_hor_za64(1, c, p, B + (c+8)*incColB + l);
        }
        for (long r=0; r<8; ++r) {
            svst1_ver_za64(0, r, p, buffer + 16*r);
            svst1_ver_za64(1, r, p, buffer + 16*r + 8);
        }
        buffer += 8*16;
    }
    if (l<k) {
        const svbool_t pk = svwhilelt_b64_s64(int64_t(l), int64_t(k));

        for (long c=0; c<8; ++c) {
            if (c  <nr) svld1_hor_za64(0, c, pk, B + (c  )*incColB + l);
            if (c+8<nr) svld1_hor_za64(1, c, pk, B + (c+8)*incColB + l);
        }
        for (long r=0; r<k-l; ++r) {
            svst1_ver_za64(0, r, p, buffer + 16*r);
            svst1_ver_za64(1, r, p, buffer + 16*r + 8);
        }
    }
}

__attribute__((noinline)) __arm_new("za") __arm_locally_streaming
void
new_pack_A_rowmajor(long mr, long k, const double *A, long incRowA,
                    double *buffer)
{
    const svbool_t p = svptrue_b64();
    long           j = 0;

    svzero_za();
    for (; j+8<=k; j+=8) {
        for (long c=0; c<8; ++c) {
            if (c   <mr) svld1_hor_za64(0, c, p, A + (c   )*incRowA + j);
            if (c+ 8<mr) svld1_hor_za64(1, c, p, A + (c+ 8)*incRowA + j);
            if (c+16<mr) svld1_hor_za64(2, c, p, A + (c+16)*incRowA + j);
            if (c+24<mr) svld1_hor_za64(3, c, p, A + (c+24)*incRowA + j);
        }
        for (long r=0; r<8; ++r) {
            svst1_ver_za64(0, r, p, buffer + 32*r);
            svst1_ver_za64(1, r, p, buffer + 32*r + 8);
            svst1_ver_za64(2, r, p, buffer + 32*r + 16);
            svst1_ver_za64(3, r, p, buffer + 32*r + 24);
        }
        buffer += 8*32;
    }
    if (j<k) {
        const svbool_t pk = svwhilelt_b64_s64(int64_t(j), int64_t(k));

        for (long c=0; c<8; ++c) {
            if (c   <mr) svld1_hor_za64(0, c, pk, A + (c   )*incRowA + j);
            if (c+ 8<mr) svld1_hor_za64(1, c, pk, A + (c+ 8)*incRowA + j);
            if (c+16<mr) svld1_hor_za64(2, c, pk, A + (c+16)*incRowA + j);
            if (c+24<mr) svld1_hor_za64(3, c, pk, A + (c+24)*incRowA + j);
        }
        for (long r=0; r<k-j; ++r) {
            svst1_ver_za64(0, r, p, buffer + 32*r);
            svst1_ver_za64(1, r, p, buffer + 32*r + 8);
            svst1_ver_za64(2, r, p, buffer + 32*r + 16);
            svst1_ver_za64(3, r, p, buffer + 32*r + 24);
        }
    }
}

//----------------------------------------------------------------------------
//  Consumer: reads the packed B panel like ugemm_32x16 (16 values per k)
//----------------------------------------------------------------------------

__attribute__((noinline)) __arm_new("za") __arm_locally_streaming
void
consume(long k, const double *Apanel, const double *Bpanel, double *out)
{
    const svbool_t  p  = svptrue_b64();
    const svcount_t pc = svptrue_c64();

    svzero_za();
    for (long l=0; l<k; ++l) {
        svfloat64x4_t a = svld1_f64_x4(pc, Apanel + 32*l);
        svfloat64x2_t b = svld1_f64_x2(pc, Bpanel + 16*l);
        svmopa_za64_f64_m(0, p, p, svget2(b, 0), svget4(a, 0));
        svmopa_za64_f64_m(1, p, p, svget2(b, 0), svget4(a, 1));
        svmopa_za64_f64_m(2, p, p, svget2(b, 0), svget4(a, 2));
        svmopa_za64_f64_m(3, p, p, svget2(b, 0), svget4(a, 3));
        svmopa_za64_f64_m(4, p, p, svget2(b, 1), svget4(a, 0));
        svmopa_za64_f64_m(5, p, p, svget2(b, 1), svget4(a, 1));
        svmopa_za64_f64_m(6, p, p, svget2(b, 1), svget4(a, 2));
        svmopa_za64_f64_m(7, p, p, svget2(b, 1), svget4(a, 3));
    }
    svst1_hor_za64(0, 0, p, out);
}

//----------------------------------------------------------------------------
//  Mechanism: rewrite only the tail rows (rows l0 .. k-1 of the panel,
//  16 doubles each) at the same addresses with one kind of store.
//----------------------------------------------------------------------------

__attribute__((noinline)) __arm_locally_streaming
void
tail_gpr(long rows, long nr, double *dst)
{
    // nr values (bit patterns) and 16 - nr zeros per row, integer stores
    for (long r=0; r<rows; ++r, dst+=16) {
        for (long j=0; j<16; ++j) {
            uint64_t v = (j<nr) ? 0x3ff0000000000000ull : 0;
            __asm__ volatile("str %x0, [%1]" :: "r"(v), "r"(dst+j) : "memory");
        }
    }
}

__attribute__((noinline)) __arm_locally_streaming
void
tail_fpr(long rows, long nr, double *dst)
{
    for (long r=0; r<rows; ++r, dst+=16) {
        for (long j=0; j<16; ++j) {
            double v = (j<nr) ? 1.0 : 0.0;
            __asm__ volatile("str %d0, [%1]" :: "w"(v), "r"(dst+j) : "memory");
        }
    }
}

__attribute__((noinline)) __arm_locally_streaming
void
tail_sve(long rows, long nr, double *dst)
{
    const svcount_t pc = svptrue_c64();
    const svbool_t  p0 = svwhilelt_b64_s64(int64_t(0), int64_t(nr));
    const svbool_t  p1 = svwhilelt_b64_s64(int64_t(8), int64_t(nr));
    const svfloat64_t one  = svdup_f64(1.0);
    for (long r=0; r<rows; ++r, dst+=16) {
        svfloat64_t z0 = svsel(p0, one, svdup_f64(0.0));
        svfloat64_t z1 = svsel(p1, one, svdup_f64(0.0));
        svst1_f64_x2(pc, dst, svcreate2(z0, z1));
    }
}

//----------------------------------------------------------------------------

static double
median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

int
main()
{
    const long ld = 64;            // B column major, 64 x 64, ld 64
    double *B   = static_cast<double *>(aligned_alloc(256, 8*ld*64));
    double *Ap  = static_cast<double *>(aligned_alloc(256, 8*32*80));
    double *b0  = static_cast<double *>(aligned_alloc(256, 8*16*80));
    double *b1  = static_cast<double *>(aligned_alloc(256, 8*16*80));
    double *a0  = static_cast<double *>(aligned_alloc(256, 8*32*80));
    double *a1  = static_cast<double *>(aligned_alloc(256, 8*32*80));
    double  out[8];
    for (long i=0; i<ld*64; ++i) B[i] = 1.0 + (i % 97)*0.01;
    for (long i=0; i<32*80; ++i) Ap[i] = 0.5;

    // 1. layout
    long bad = 0, cases = 0;
    for (long nr=1; nr<=16; ++nr) {
        for (long k=1; k<=40; ++k) {
            std::fill(b0, b0 + 16*80, 7.0);
            std::fill(b1, b1 + 16*80, 7.0);
            old_pack_B_colmajor(nr, k, B, ld, b0);
            new_pack_B_colmajor(nr, k, B, ld, b1);
            bad += std::memcmp(b0, b1, 8*16*80)!=0;
            ++cases;
        }
    }
    for (long mr=1; mr<=32; ++mr) {
        for (long k=1; k<=40; ++k) {
            std::fill(a0, a0 + 32*80, 7.0);
            std::fill(a1, a1 + 32*80, 7.0);
            // A row major: mr rows, row i at B + i*ld (k <= 40 <= ld)
            old_pack_A_rowmajor(mr, k, B, ld, a0);
            new_pack_A_rowmajor(mr, k, B, ld, a1);
            bad += std::memcmp(a0, a1, 8*32*80)!=0;
            ++cases;
        }
    }
    printf("layout: %ld cases (pack_B nr 1..16, pack_A mr 1..32, k 1..40, "
           "incl. 7.0 sentinels after the panel), %ld differ\n", cases, bad);

    // 2. timing of pack + consume, ns per panel
    const long REP = 20000;
    printf("\npack_B_colmajor + kernel-like read (ns per panel, median of 5)\n");
    printf("  nr    k   old     new    old/new\n");
    for (long nr : {8, 12, 15, 16}) {
        for (long k : {8, 12, 15, 16, 64, 68}) {
            std::vector<double> to, tn;
            for (int round=0; round<5; ++round) {
                for (int v=0; v<2; ++v) {
                    double t0 = now();
                    for (long r=0; r<REP; ++r) {
                        if (v==0) old_pack_B_colmajor(nr, k, B, ld, b0);
                        else      new_pack_B_colmajor(nr, k, B, ld, b0);
                        consume(k, Ap, b0, out);
                    }
                    (v==0 ? to : tn).push_back(1e9*(now()-t0)/REP);
                }
            }
            printf("  %2ld  %3ld  %6.1f  %6.1f  %5.2f\n", nr, k, median(to),
                   median(tn), median(to)/median(tn));
        }
    }

    // 3. mechanism: same buffer and addresses, only the tail rows rewritten
    printf("\nmechanism: new pack, then the %s rewritten, then read "
           "(ns per panel, median of 5)\n", "tail rows (k mod 8 of them)");
    printf("  nr    k   none    gpr     fpr     sve\n");
    for (long nr : {12, 16}) {
        for (long k : {12, 15, 68}) {
            long l0 = k - k%8, rows = k%8;
            std::vector<double> t[4];
            for (int round=0; round<5; ++round) {
                for (int v=0; v<4; ++v) {
                    double t0 = now();
                    for (long r=0; r<REP; ++r) {
                        new_pack_B_colmajor(nr, k, B, ld, b0);
                        if (v==1) tail_gpr(rows, nr, b0 + 16*l0);
                        if (v==2) tail_fpr(rows, nr, b0 + 16*l0);
                        if (v==3) tail_sve(rows, nr, b0 + 16*l0);
                        consume(k, Ap, b0, out);
                    }
                    t[v].push_back(1e9*(now()-t0)/REP);
                }
            }
            printf("  %2ld  %3ld  %6.1f  %6.1f  %6.1f  %6.1f\n", nr, k,
                   median(t[0]), median(t[1]), median(t[2]), median(t[3]));
        }
    }
    printf("(checksum %g)\n", out[0]);
    return 0;
}
