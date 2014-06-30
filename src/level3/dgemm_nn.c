#include <ulmblas.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>

#define MC  384
#define KC  384
#define NC  4096

#define MR  4
#define NR  4

//
//  Local buffers for storing panels from A, B and C
//
static double _A[MC*KC];
static double _B[KC*NC];
static double _C[MR*NR];

//
//  Packing complete panels from A (i.e. without padding)
//
static void
pack_MRxk(int k, const double *A, int incRowA, int incColA,
          double *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

//
//  Packing panels from A with padding if required
//
static void
pack_A(int mc, int kc, const double *A, int incRowA, int incColA,
       double *buffer)
{
    int mp  = mc / MR;
    int _mr = mc % MR;

    int i, j;

    for (i=0; i<mp; ++i) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    if (_mr>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<MR; ++i) {
                buffer[i] = 0.0;
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR(int k, const double *B, int incRowB, int incColB,
          double *buffer)
{
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

//
//  Packing panels from B with padding if required
//
static void
pack_B(int kc, int nc, const double *B, int incRowB, int incColB,
       double *buffer)
{
    int np  = nc / NR;
    int _nr = nc % NR;

    int i, j;

    for (j=0; j<np; ++j) {
        pack_kxNR(kc, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    if (_nr>0) {
        for (i=0; i<kc; ++i) {
            for (j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (j=_nr; j<NR; ++j) {
                buffer[j] = 0.0;
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

//
//  Micro kernel for multiplying panels from A and B.  With nextA, nextB
//  we give the possibility of prefetching.
//
#if (MR==4) && (NR==4)


static void
dgemm_micro_kernel(long kc,
                   double alpha, const double *A, const double *B,
                   double beta,
                   double *C, int incRowC, long incColC,
                   const double *nextA, const double *nextB)
{
    long kb = kc / 4;
    long kl = kc % 4;

    __asm__ volatile
    (
    " movq         %2,             %%rax     \n\t" // load address of A
    " movq         %3,             %%rbx     \n\t" // load address of B
    " movq         %4,             %%rcx     \n\t" // load address of C
    " movq         %5,             %%rdi     \n\t" // load incColC
    "                                        \n\t"
    "movq          %7,             %%r9      \n\t" // load address of nextB
    "prefetcht2    (%%r9)                    \n\t" // prefetch nextB
    "                                        \n\t"
    " xorpd        %%xmm3,         %%xmm3    \n\t"
    " xorpd        %%xmm4,         %%xmm4    \n\t"
    " xorpd        %%xmm5,         %%xmm5    \n\t"
    " xorpd        %%xmm6,         %%xmm6    \n\t"
    "                                        \n\t"
    " xorpd        %%xmm8,         %%xmm8    \n\t"
    " xorpd        %%xmm9,         %%xmm9    \n\t"
    " xorpd        %%xmm10,        %%xmm10   \n\t"
    " xorpd        %%xmm11,        %%xmm11   \n\t"
    " xorpd        %%xmm12,        %%xmm12   \n\t"
    " xorpd        %%xmm13,        %%xmm13   \n\t"
    " xorpd        %%xmm14,        %%xmm14   \n\t"
    " xorpd        %%xmm15,        %%xmm15   \n\t"
    "                                        \n\t"
    " movaps       (%%rax),        %%xmm0    \n\t" // load (a[0],a[1])
    " movaps     16(%%rax),        %%xmm1    \n\t" // load (a[2],a[3])
    "                                        \n\t"
    " movaps       (%%rbx),        %%xmm2    \n\t" // load (b[0],b[1])
    "                                        \n\t"
    " movq         %0,             %%rsi     \n\t" // i = kb
    " testq        %%rsi,          %%rsi     \n\t" // check i via logical AND.
    " je           .DINITLOOPKL              \n\t" // if i == 0 skip unrolled
    "                                        \n\t" // loop
    ".DLOOPKB:                               \n\t"
    "                                        \n\t"
    // k = 0
    "                                        \n\t"
    " addpd        %%xmm3,         %%xmm11   \n\t" // update ab_02_13
    " movaps     16(%%rbx),        %%xmm3    \n\t" // load (b[2],b[3])
    " addpd        %%xmm4,         %%xmm15   \n\t" // update ab_22_33
    "                                        \n\t"
    // compute diag pairs (a[0]*b[0],a[1]*b[1]) and (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " movaps       %%xmm2,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm2,         %%xmm7    \n\t" // swap -> (b[1],b[0])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm2    \n\t" // (a[0]*b[0],a[1]*b[1])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " addpd        %%xmm5,         %%xmm10   \n\t" // update ab_03_12
    " addpd        %%xmm6,         %%xmm14   \n\t" // update ab_22_33
    // compute diag pairs (a[0]*b[1],a[1]*b[0]) and (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " movaps       %%xmm7,         %%xmm6    \n\t"
    " mulpd        %%xmm0,         %%xmm7    \n\t" // (a[0]*b[1],a[1]*b[0])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " addpd        %%xmm2,         %%xmm9    \n\t" // update ab_00_11
    " movaps     32(%%rbx),        %%xmm2    \n\t" // load *next* (b[0],b[1])
    " addpd        %%xmm4,         %%xmm13   \n\t" // update ab_20_31
    "                                        \n\t"
    // compute diag pairs (a[0]*b[2],a[1]*b[3]) and (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " movaps       %%xmm3,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm3,         %%xmm5    \n\t" // swap -> (b[3],b[2])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm3    \n\t" // (a[0]*b[2],a[1]*b[3])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " addpd        %%xmm7,         %%xmm8    \n\t" // update ab_01_10
    " addpd        %%xmm6,         %%xmm12   \n\t" // update ab_21_30
    "                                        \n\t"
    // compute diag pairs (a[0]*b[3],a[1]*b[2]) and (a[2]*b[3],a[3]*b[2])
    "                                        \n\t"
    " movaps       %%xmm5,         %%xmm6    \n\t"
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm5    \n\t" // (a[0]*b[3],a[1]*b[2])
    " movaps     32(%%rax),        %%xmm0    \n\t" // load *next* (a[0],a[1])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[3],a[3]*b[2])
    " movaps     48(%%rax),        %%xmm1    \n\t" // load *next* (a[2],a[3])
    "                                        \n\t"
    " addq         $4 * 8,         %%rax     \n\t" // A += 4
    " addq         $4 * 8,         %%rbx     \n\t" // B += 4
    "                                        \n\t"
    // k = 1
    "                                        \n\t"
    " addpd        %%xmm3,         %%xmm11   \n\t" // update ab_02_13
    " movaps     16(%%rbx),        %%xmm3    \n\t" // load (b[2],b[3])
    " addpd        %%xmm4,         %%xmm15   \n\t" // update ab_22_33
    "                                        \n\t"
    // compute diag pairs (a[0]*b[0],a[1]*b[1]) and (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " movaps       %%xmm2,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm2,         %%xmm7    \n\t" // swap -> (b[1],b[0])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm2    \n\t" // (a[0]*b[0],a[1]*b[1])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " addpd        %%xmm5,         %%xmm10   \n\t" // update ab_03_12
    " addpd        %%xmm6,         %%xmm14   \n\t" // update ab_22_33
    // compute diag pairs (a[0]*b[1],a[1]*b[0]) and (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " movaps       %%xmm7,         %%xmm6    \n\t"
    " mulpd        %%xmm0,         %%xmm7    \n\t" // (a[0]*b[1],a[1]*b[0])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " addpd        %%xmm2,         %%xmm9    \n\t" // update ab_00_11
    " movaps     32(%%rbx),        %%xmm2    \n\t" // load *next* (b[0],b[1])
    " addpd        %%xmm4,         %%xmm13   \n\t" // update ab_20_31
    "                                        \n\t"
    // compute diag pairs (a[0]*b[2],a[1]*b[3]) and (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " movaps       %%xmm3,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm3,         %%xmm5    \n\t" // swap -> (b[3],b[2])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm3    \n\t" // (a[0]*b[2],a[1]*b[3])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " addpd        %%xmm7,         %%xmm8    \n\t" // update ab_01_10
    " addpd        %%xmm6,         %%xmm12   \n\t" // update ab_21_30
    "                                        \n\t"
    // compute diag pairs (a[0]*b[3],a[1]*b[2]) and (a[2]*b[3],a[3]*b[2])
    "                                        \n\t"
    " movaps       %%xmm5,         %%xmm6    \n\t"
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm5    \n\t" // (a[0]*b[3],a[1]*b[2])
    " movaps     32(%%rax),        %%xmm0    \n\t" // load *next* (a[0],a[1])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[3],a[3]*b[2])
    " movaps     48(%%rax),        %%xmm1    \n\t" // load *next* (a[2],a[3])
    "                                        \n\t"
    " addq         $4 * 8,         %%rax     \n\t" // A += 4
    " addq         $4 * 8,         %%rbx     \n\t" // B += 4
    "                                        \n\t"
    // k = 2
    "                                        \n\t"
    " addpd        %%xmm3,         %%xmm11   \n\t" // update ab_02_13
    " movaps     16(%%rbx),        %%xmm3    \n\t" // load (b[2],b[3])
    " addpd        %%xmm4,         %%xmm15   \n\t" // update ab_22_33
    "                                        \n\t"
    // compute diag pairs (a[0]*b[0],a[1]*b[1]) and (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " movaps       %%xmm2,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm2,         %%xmm7    \n\t" // swap -> (b[1],b[0])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm2    \n\t" // (a[0]*b[0],a[1]*b[1])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " addpd        %%xmm5,         %%xmm10   \n\t" // update ab_03_12
    " addpd        %%xmm6,         %%xmm14   \n\t" // update ab_22_33
    // compute diag pairs (a[0]*b[1],a[1]*b[0]) and (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " movaps       %%xmm7,         %%xmm6    \n\t"
    " mulpd        %%xmm0,         %%xmm7    \n\t" // (a[0]*b[1],a[1]*b[0])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " addpd        %%xmm2,         %%xmm9    \n\t" // update ab_00_11
    " movaps     32(%%rbx),        %%xmm2    \n\t" // load *next* (b[0],b[1])
    " addpd        %%xmm4,         %%xmm13   \n\t" // update ab_20_31
    "                                        \n\t"
    // compute diag pairs (a[0]*b[2],a[1]*b[3]) and (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " movaps       %%xmm3,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm3,         %%xmm5    \n\t" // swap -> (b[3],b[2])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm3    \n\t" // (a[0]*b[2],a[1]*b[3])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " addpd        %%xmm7,         %%xmm8    \n\t" // update ab_01_10
    " addpd        %%xmm6,         %%xmm12   \n\t" // update ab_21_30
    "                                        \n\t"
    // compute diag pairs (a[0]*b[3],a[1]*b[2]) and (a[2]*b[3],a[3]*b[2])
    "                                        \n\t"
    " movaps       %%xmm5,         %%xmm6    \n\t"
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm5    \n\t" // (a[0]*b[3],a[1]*b[2])
    " movaps     32(%%rax),        %%xmm0    \n\t" // load *next* (a[0],a[1])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[3],a[3]*b[2])
    " movaps     48(%%rax),        %%xmm1    \n\t" // load *next* (a[2],a[3])
    "                                        \n\t"
    " addq         $4 * 8,         %%rax     \n\t" // A += 4
    " addq         $4 * 8,         %%rbx     \n\t" // B += 4
    "                                        \n\t"
    // k = 3
    "                                        \n\t"
    " addpd        %%xmm3,         %%xmm11   \n\t" // update ab_02_13
    " movaps     16(%%rbx),        %%xmm3    \n\t" // load (b[2],b[3])
    " addpd        %%xmm4,         %%xmm15   \n\t" // update ab_22_33
    "                                        \n\t"
    // compute diag pairs (a[0]*b[0],a[1]*b[1]) and (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " movaps       %%xmm2,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm2,         %%xmm7    \n\t" // swap -> (b[1],b[0])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm2    \n\t" // (a[0]*b[0],a[1]*b[1])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " addpd        %%xmm5,         %%xmm10   \n\t" // update ab_03_12
    " addpd        %%xmm6,         %%xmm14   \n\t" // update ab_22_33
    // compute diag pairs (a[0]*b[1],a[1]*b[0]) and (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " movaps       %%xmm7,         %%xmm6    \n\t"
    " mulpd        %%xmm0,         %%xmm7    \n\t" // (a[0]*b[1],a[1]*b[0])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " addpd        %%xmm2,         %%xmm9    \n\t" // update ab_00_11
    " movaps     32(%%rbx),        %%xmm2    \n\t" // load *next* (b[0],b[1])
    " addpd        %%xmm4,         %%xmm13   \n\t" // update ab_20_31
    "                                        \n\t"
    // compute diag pairs (a[0]*b[2],a[1]*b[3]) and (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " movaps       %%xmm3,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm3,         %%xmm5    \n\t" // swap -> (b[3],b[2])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm3    \n\t" // (a[0]*b[2],a[1]*b[3])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " addpd        %%xmm7,         %%xmm8    \n\t" // update ab_01_10
    " addpd        %%xmm6,         %%xmm12   \n\t" // update ab_21_30
    "                                        \n\t"
    // compute diag pairs (a[0]*b[3],a[1]*b[2]) and (a[2]*b[3],a[3]*b[2])
    "                                        \n\t"
    " movaps       %%xmm5,         %%xmm6    \n\t"
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm5    \n\t" // (a[0]*b[3],a[1]*b[2])
    " movaps     32(%%rax),        %%xmm0    \n\t" // load *next* (a[0],a[1])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[3],a[3]*b[2])
    " movaps     48(%%rax),        %%xmm1    \n\t" // load *next* (a[2],a[3])
    "                                        \n\t"
    " addq         $4 * 8,         %%rax     \n\t" // A += 4
    " addq         $4 * 8,         %%rbx     \n\t" // B += 4
    "                                        \n\t"
    " decq         %%rsi                     \n\t" // --i
    " jne          .DLOOPKB                  \n\t" // iterate again if i != 0.
    "                                        \n\t"
    ".DINITLOOPKL:                           \n\t"
    "                                        \n\t"
    " movq         %1,             %%rsi     \n\t" // i = kl
    " testq        %%rsi,          %%rsi     \n\t" // check i via logical AND.
    " je           .DDONE                    \n\t" // if i == 0 we are done
    "                                        \n\t"
    ".DLOOPKL:                               \n\t"
    "                                        \n\t"
    " addpd        %%xmm3,         %%xmm11   \n\t" // update ab_02_13
    " movaps     16(%%rbx),        %%xmm3    \n\t" // load (b[2],b[3])
    " addpd        %%xmm4,         %%xmm15   \n\t" // update ab_22_33
    "                                        \n\t"
    // compute diag pairs (a[0]*b[0],a[1]*b[1]) and (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " movaps       %%xmm2,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm2,         %%xmm7    \n\t" // swap -> (b[1],b[0])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm2    \n\t" // (a[0]*b[0],a[1]*b[1])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[0],a[3]*b[1])
    "                                        \n\t"
    " addpd        %%xmm5,         %%xmm10   \n\t" // update ab_03_12
    " addpd        %%xmm6,         %%xmm14   \n\t" // update ab_22_33
    // compute diag pairs (a[0]*b[1],a[1]*b[0]) and (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " movaps       %%xmm7,         %%xmm6    \n\t"
    " mulpd        %%xmm0,         %%xmm7    \n\t" // (a[0]*b[1],a[1]*b[0])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[1],a[3]*b[0])
    "                                        \n\t"
    " addpd        %%xmm2,         %%xmm9    \n\t" // update ab_00_11
    " movaps     32(%%rbx),        %%xmm2    \n\t" // load *next* (b[0],b[1])
    " addpd        %%xmm4,         %%xmm13   \n\t" // update ab_20_31
    "                                        \n\t"
    // compute diag pairs (a[0]*b[2],a[1]*b[3]) and (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " movaps       %%xmm3,         %%xmm4    \n\t"
    " pshufd $0x4e,%%xmm3,         %%xmm5    \n\t" // swap -> (b[3],b[2])
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm3    \n\t" // (a[0]*b[2],a[1]*b[3])
    " mulpd        %%xmm1,         %%xmm4    \n\t" // (a[2]*b[2],a[3]*b[3])
    "                                        \n\t"
    " addpd        %%xmm7,         %%xmm8    \n\t" // update ab_01_10
    " addpd        %%xmm6,         %%xmm12   \n\t" // update ab_21_30
    "                                        \n\t"
    // compute diag pairs (a[0]*b[3],a[1]*b[2]) and (a[2]*b[3],a[3]*b[2])
    "                                        \n\t"
    " movaps       %%xmm5,         %%xmm6    \n\t"
    "                                        \n\t"
    " mulpd        %%xmm0,         %%xmm5    \n\t" // (a[0]*b[3],a[1]*b[2])
    " movaps     32(%%rax),        %%xmm0    \n\t" // load *next* (a[0],a[1])
    " mulpd        %%xmm1,         %%xmm6    \n\t" // (a[2]*b[3],a[3]*b[2])
    " movaps     48(%%rax),        %%xmm1    \n\t" // load *next* (a[2],a[3])
    "                                        \n\t"
    " addq         $4 * 8,         %%rax     \n\t" // A += 4
    " addq         $4 * 8,         %%rbx     \n\t" // B += 4
    "                                        \n\t"
        "prefetcht2        0 * 8(%%r9)   \n\t" // prefetch b_next[0]
        "prefetcht2        8 * 8(%%r9)   \n\t" // prefetch b_next[8]
    " decq         %%rsi                     \n\t" // --i
    " jne          .DLOOPKL                  \n\t" // iterate again if i != 0.
    "                                        \n\t"
    ".DDONE:                                 \n\t"
    "                                        \n\t"
    " addpd        %%xmm3,         %%xmm11   \n\t" // update ab_02_13
    " addpd        %%xmm4,         %%xmm15   \n\t" // update ab_22_33
    "                                        \n\t"
    " addpd        %%xmm5,         %%xmm10   \n\t" // update ab_03_12
    " addpd        %%xmm6,         %%xmm14   \n\t" // update ab_22_33
    "                                        \n\t"
    /*
    " movlpd       %%xmm9,         0*8(%%rcx)\n\t" // copy ab_00
    " movhpd       %%xmm8,         1*8(%%rcx)\n\t" // copy ab_10
    " movlpd       %%xmm13,        2*8(%%rcx)\n\t" // copy ab_20
    " movhpd       %%xmm12,        3*8(%%rcx)\n\t" // copy ab_30
    "                                        \n\t"
    " movlpd       %%xmm8,  (0*8+1*32)(%%rcx)\n\t" // copy ab_01
    " movhpd       %%xmm9,  (1*8+1*32)(%%rcx)\n\t" // copy ab_11
    " movlpd       %%xmm12, (2*8+1*32)(%%rcx)\n\t" // copy ab_21
    " movhpd       %%xmm13, (3*8+1*32)(%%rcx)\n\t" // copy ab_31
    "                                        \n\t"
    " movlpd       %%xmm11, (0*8+2*32)(%%rcx)\n\t" // copy ab_02
    " movhpd       %%xmm10, (1*8+2*32)(%%rcx)\n\t" // copy ab_12
    " movlpd       %%xmm15, (2*8+2*32)(%%rcx)\n\t" // copy ab_22
    " movhpd       %%xmm14, (3*8+2*32)(%%rcx)\n\t" // copy ab_32
    "                                        \n\t"
    " movlpd       %%xmm10, (0*8+3*32)(%%rcx)\n\t" // copy ab_03
    " movhpd       %%xmm11, (1*8+3*32)(%%rcx)\n\t" // copy ab_13
    " movlpd       %%xmm14, (2*8+3*32)(%%rcx)\n\t" // copy ab_23
    " movhpd       %%xmm15, (3*8+3*32)(%%rcx)\n\t" // copy ab_33
    */
    "                                        \n\t"
    " movaps       %%xmm8,          %%xmm0   \n\t"
    " movsd        %%xmm9,          %%xmm8   \n\t"
    " movsd        %%xmm0,          %%xmm9   \n\t"
    "                                        \n\t"
    " movaps       %%xmm10,         %%xmm0   \n\t"
    " movsd        %%xmm11,         %%xmm10  \n\t"
    " movsd        %%xmm0,          %%xmm11  \n\t"
    "                                        \n\t"
    " movaps       %%xmm12,         %%xmm0   \n\t"
    " movsd        %%xmm13,         %%xmm12  \n\t"
    " movsd        %%xmm0,          %%xmm13  \n\t"
    "                                        \n\t"
    " movaps       %%xmm14,         %%xmm0   \n\t"
    " movsd        %%xmm15,         %%xmm14  \n\t"
    " movsd        %%xmm0,          %%xmm15  \n\t"
    "                                        \n\t"
    /*
    // copy to AB
    " movaps       %%xmm8,          (%%rcx)  \n\t"
    " movaps       %%xmm12,       16(%%rcx)  \n\t"
    "                                        \n\t"
    " movaps       %%xmm9,        32(%%rcx)  \n\t"
    " movaps       %%xmm13,       48(%%rcx)  \n\t"
    "                                        \n\t"
    " movaps       %%xmm10,       64(%%rcx)  \n\t"
    " movaps       %%xmm14,       80(%%rcx)  \n\t"
    "                                        \n\t"
    " movaps       %%xmm11,       96(%%rcx)  \n\t"
    " movaps       %%xmm15,      112(%%rcx)  \n\t"
    */
    "                                        \n\t"
    " leaq         (,%%rdi,8),      %%rdi    \n\t" // incColC *= sizeof(double)
    "                                        \n\t"
    " movaps       (%%rcx),         %%xmm0   \n\t" // load c00 and c10,
    " addpd        %%xmm8,          %%xmm0   \n\t" // add  ab00 and ab10
    " movaps       %%xmm0,          (%%rcx)  \n\t" // store c00 and c10,
    "                                        \n\t"
    " movaps     16(%%rcx),         %%xmm1   \n\t" // load c20 and c30,
    " addpd        %%xmm12,         %%xmm1   \n\t" // add  ab20 and ab30
    " movaps       %%xmm1,        16(%%rcx)  \n\t" // store c20 and c30,
    "                                        \n\t"
    " addq         %%rdi,           %%rcx    \n\t" // next col of C
    "                                        \n\t"
    " movaps       (%%rcx),         %%xmm0   \n\t" // load c01 and c11,
    " addpd        %%xmm9,          %%xmm0   \n\t" // add  ab01 and ab11
    " movaps       %%xmm0,          (%%rcx)  \n\t" // store c01 and c11,
    "                                        \n\t"
    " movaps     16(%%rcx),         %%xmm1   \n\t" // load c21 and c31,
    " addpd        %%xmm13,         %%xmm1   \n\t" // add  ab21 and ab31
    " movaps       %%xmm1,        16(%%rcx)  \n\t" // store c21 and c31,
    "                                        \n\t"
    " addq         %%rdi,           %%rcx    \n\t" // next col of C
    "                                        \n\t"
    " movaps       (%%rcx),         %%xmm0   \n\t" // load c02 and c12,
    " addpd        %%xmm10,         %%xmm0   \n\t" // add  ab02 and ab12
    " movaps       %%xmm0,          (%%rcx)  \n\t" // store c02 and c12,
    "                                        \n\t"
    " movaps     16(%%rcx),         %%xmm1   \n\t" // load c22 and c32,
    " addpd        %%xmm14,         %%xmm1   \n\t" // add  ab22 and ab32
    " movaps       %%xmm1,        16(%%rcx)  \n\t" // store c22 and c32,
    "                                        \n\t"
    " addq         %%rdi,           %%rcx    \n\t" // next col of C
    "                                        \n\t"
    " movaps       (%%rcx),         %%xmm0   \n\t" // load c03 and c13,
    " addpd        %%xmm11,         %%xmm0   \n\t" // add  ab03 and ab13
    " movaps       %%xmm0,          (%%rcx)  \n\t" // store c03 and c13,
    "                                        \n\t"
    " movaps     16(%%rcx),         %%xmm1   \n\t" // load c23 and c33,
    " addpd        %%xmm15,         %%xmm1   \n\t" // add  ab23 and ab33
    " movaps       %%xmm1,        16(%%rcx)  \n\t" // store c23 and c33,
    :  // output
    :  // input
        "m" (kb),       // 0
        "m" (kl),       // 1
        "m" (A),        // 2
        "m" (B),        // 3
        "m" (C),        // 4
        "m" (incColC),  // 5
        "m" (nextA),    // 6
        "m" (nextB)     // 7
    :  // register
        "rax", "rbx", "rcx", "rsi", "r9", "rdi",
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
        "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
        "memory"
    );

    /*
    if (beta==0.0) {
        for (j=0; j<4; ++j) {
            for (i=0; i<4; ++i) {
                C[i*incRowC+j*incColC] = 0.0;
            }
        }
    } else {
        for (j=0; j<4; ++j) {
            for (i=0; i<4; ++i) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

    if (alpha==1.0) {
        for (j=0; j<4; ++j) {
            for (i=0; i<4; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*4];
            }
        }
    } else {
        for (j=0; j<4; ++j) {
            for (i=0; i<4; ++i) {
                C[i*incRowC+j*incColC] += alpha*AB[i+j*4];
            }
        }
    }*/

    /*
    if (beta==0.0) {
        if (alpha==1.0) {
            for (j=0; j<4; ++j) {
                for (i=0; i<4; ++i) {
                    C[i*incRowC+j*incColC] = AB[i+j*4];
                }
            }
        } else {
            for (j=0; j<4; ++j) {
                for (i=0; i<4; ++i) {
                    C[i*incRowC+j*incColC] = alpha*AB[i+j*4];
                }
            }
        }
    } else if (beta==1.0) {
        if (alpha==1.0) {
            for (j=0; j<4; ++j) {
                for (i=0; i<4; ++i) {
                    C[i*incRowC+j*incColC] += AB[i+j*4];
                }
            }
        } else {
            for (j=0; j<4; ++j) {
                for (i=0; i<4; ++i) {
                    C[i*incRowC+j*incColC] += alpha*AB[i+j*4];
                }
            }
        }
     } else {
        for (j=0; j<4; ++j) {
            for (i=0; i<4; ++i) {
                C[i*incRowC+j*incColC] = beta*C[i*incRowC+j*incColC]
                                       + alpha*AB[i+j*4];
            }
        }
    }
    */
}
#else
#error "In this branch we use a hard coded micro kernel.  It requires MR=NR=4\n"
#endif

//
//  Compute Y += alpha*X
//
static void
dgeaxpy(int           m,
        int           n,
        double        alpha,
        const double  *X,
        int           incRowX,
        int           incColX,
        double        *Y,
        int           incRowY,
        int           incColY)
{
    int i, j;


    if (alpha!=1.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
            }
        }
    }
}

//
//  Compute X *= alpha
//
static void
dgescal(int     m,
        int     n,
        double  alpha,
        double  *X,
        int     incRowX,
        int     incColX)
{
    int i, j;

    if (alpha!=0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] *= alpha;
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = 0.0;
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
static void
dgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   double  alpha,
                   double  beta,
                   double  *C,
                   int     incRowC,
                   int     incColC)
{
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;

    const double *nextA;
    const double *nextB;

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;
        nextB = &_B[j*kc*NR];

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;
            nextA = &_A[(i+1)*kc*MR];

            if (i==mp-1) {
                nextA = _A;
                nextB = &_B[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = _B;
                }
            }

            if (mr==MR && nr==NR) {
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                   beta,
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC,
                                   nextA, nextB);
            } else {
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                   0.0,
                                   _C, 1, MR,
                                   nextA, nextB);
                dgescal(mr, nr, beta,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
                dgeaxpy(mr, nr, 1.0, _C, 1, MR,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}

//
//  Compute C <- beta*C + alpha*A*B
//
void
ULMBLAS(dgemm_nn)(int            m,
                  int            n,
                  int            k,
                  double         alpha,
                  const double   *A,
                  int            incRowA,
                  int            incColA,
                  const double   *B,
                  int            incRowB,
                  int            incColB,
                  double         beta,
                  double         *C,
                  int            incRowC,
                  int            incColC)
{
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    double _beta;

    if (alpha==0.0 || k==0) {
        dgescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            _beta = (l==0) ? beta : 1.0;

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   _B);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc,
                       &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                       _A);

                dgemm_macro_kernel(mc, nc, kc, alpha, _beta,
                                   &C[i*MC*incRowC+j*NC*incColC],
                                   incRowC, incColC);
            }
        }
    }
}
