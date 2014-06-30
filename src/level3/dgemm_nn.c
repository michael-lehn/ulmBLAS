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
                   double *C, long incRowC, long incColC,
                   const double *nextA, const double *nextB)
{
    long kb = kc / 4;
    long kl = kc % 4;

    __asm__ volatile
    (
        "                                \n\t"
        "                                \n\t"
        "movq          %2, %%rax         \n\t" // load address of a.
        "movq          %3, %%rbx         \n\t" // load address of b.
        "movq          %9, %%r9          \n\t" // load address of b_next.
        "movq         %10, %%r11         \n\t" // load address of a_next.
        "                                \n\t"
        "subq    $-8 * 16, %%rax         \n\t" // increment pointers to allow byte
        "subq    $-8 * 16, %%rbx         \n\t" // offsets in the unrolled iterations.
        "                                \n\t"
        "movaps  -8 * 16(%%rax), %%xmm0  \n\t" // initialize loop by pre-loading elements
        "movaps  -7 * 16(%%rax), %%xmm1  \n\t" // of a and b.
        "movaps  -8 * 16(%%rbx), %%xmm2  \n\t"
        "                                \n\t"
        "movq          %6, %%rcx         \n\t" // load address of c
        "movq          %8, %%rdi         \n\t" // load cs_c
        "leaq        (,%%rdi,8), %%rdi   \n\t" // cs_c *= sizeof(double)
        "leaq   (%%rcx,%%rdi,2), %%r10   \n\t" // load address of c + 2*cs_c;
        "                                \n\t"
        //"prefetcht2   0 * 8(%%r9)        \n\t" // prefetch b_next
        "                                \n\t"
        "xorpd     %%xmm3,  %%xmm3       \n\t"
        "xorpd     %%xmm4,  %%xmm4       \n\t"
        "xorpd     %%xmm5,  %%xmm5       \n\t"
        "xorpd     %%xmm6,  %%xmm6       \n\t"
        "                                \n\t"
        "prefetcht2   3 * 8(%%rcx)       \n\t" // prefetch c + 0*cs_c
        "xorpd     %%xmm8,  %%xmm8       \n\t"
        "movaps    %%xmm8,  %%xmm9       \n\t"
        "prefetcht2   3 * 8(%%rcx,%%rdi) \n\t" // prefetch c + 1*cs_c
        "movaps    %%xmm8, %%xmm10       \n\t"
        "movaps    %%xmm8, %%xmm11       \n\t"
        "prefetcht2   3 * 8(%%r10)       \n\t" // prefetch c + 2*cs_c
        "movaps    %%xmm8, %%xmm12       \n\t"
        "movaps    %%xmm8, %%xmm13       \n\t"
        "prefetcht2   3 * 8(%%r10,%%rdi) \n\t" // prefetch c + 3*cs_c
        "movaps    %%xmm8, %%xmm14       \n\t"
        "movaps    %%xmm8, %%xmm15       \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "movq      %0, %%rsi             \n\t" // i = k_iter;
        "testq  %%rsi, %%rsi             \n\t" // check i via logical AND.
        "je     .DCONSIDKLEFT            \n\t" // if i == 0, jump to code that
        "                                \n\t" // contains the k_left loop.
        "                                \n\t"
        "                                \n\t"
        ".DLOOPKITER:                    \n\t" // MAIN LOOP
        "                                \n\t"
        "prefetcht0  (4*35+1) * 8(%%rax) \n\t"
        //"prefetcht0  (8*97+4) * 8(%%rax) \n\t"
        "                                \n\t"
        //"prefetcht0  67*4 * 8(%%r11)       \n\t" // prefetch a_next[0]
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t" // iteration 0
        "movaps  -7 * 16(%%rbx), %%xmm3  \n\t"
        "addpd   %%xmm4, %%xmm15         \n\t"
        "movaps  %%xmm2, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
        "mulpd   %%xmm0, %%xmm2          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "addpd   %%xmm5, %%xmm10         \n\t"
        "addpd   %%xmm6, %%xmm14         \n\t"
        "movaps  %%xmm7, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm7          \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "                                \n\t"
        "addpd   %%xmm2, %%xmm9          \n\t"
        "movaps  -6 * 16(%%rbx), %%xmm2  \n\t"
        "addpd   %%xmm4, %%xmm13         \n\t"
        "movaps  %%xmm3, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
        "mulpd   %%xmm0, %%xmm3          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "addpd   %%xmm7, %%xmm8          \n\t"
        "addpd   %%xmm6, %%xmm12         \n\t"
        "movaps  %%xmm5, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm5          \n\t"
        "movaps  -6 * 16(%%rax), %%xmm0  \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "movaps  -5 * 16(%%rax), %%xmm1  \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t" // iteration 1
        "movaps  -5 * 16(%%rbx), %%xmm3  \n\t"
        "addpd   %%xmm4, %%xmm15         \n\t"
        "movaps  %%xmm2, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
        "mulpd   %%xmm0, %%xmm2          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "addpd   %%xmm5, %%xmm10         \n\t"
        "addpd   %%xmm6, %%xmm14         \n\t"
        "movaps  %%xmm7, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm7          \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "                                \n\t"
        "addpd   %%xmm2, %%xmm9          \n\t"
        "movaps  -4 * 16(%%rbx), %%xmm2  \n\t"
        "addpd   %%xmm4, %%xmm13         \n\t"
        "movaps  %%xmm3, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
        "mulpd   %%xmm0, %%xmm3          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "addpd   %%xmm7, %%xmm8          \n\t"
        "addpd   %%xmm6, %%xmm12         \n\t"
        "movaps  %%xmm5, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm5          \n\t"
        "movaps  -4 * 16(%%rax), %%xmm0  \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "movaps  -3 * 16(%%rax), %%xmm1  \n\t"
        "                                \n\t"
        "                                \n\t"
        "prefetcht0  (4*37+1) * 8(%%rax) \n\t"
        //"prefetcht0  (8*97+12)* 8(%%rax) \n\t"
        "                                \n\t"
        //"prefetcht0  69*4 * 8(%%r11)       \n\t" // prefetch a_next[8]
        //"subq  $-4 * 4 * 8, %%r11        \n\t" // a_next += 4*4 (unroll x mr)
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t" // iteration 2
        "movaps  -3 * 16(%%rbx), %%xmm3  \n\t"
        "addpd   %%xmm4, %%xmm15         \n\t"
        "movaps  %%xmm2, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
        "mulpd   %%xmm0, %%xmm2          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "addpd   %%xmm5, %%xmm10         \n\t"
        "addpd   %%xmm6, %%xmm14         \n\t"
        "movaps  %%xmm7, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm7          \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "                                \n\t"
        "addpd   %%xmm2, %%xmm9          \n\t"
        "movaps  -2 * 16(%%rbx), %%xmm2  \n\t"
        "addpd   %%xmm4, %%xmm13         \n\t"
        "movaps  %%xmm3, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
        "mulpd   %%xmm0, %%xmm3          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "                                \n\t"
        "addpd   %%xmm7, %%xmm8          \n\t"
        "addpd   %%xmm6, %%xmm12         \n\t"
        "movaps  %%xmm5, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm5          \n\t"
        "movaps  -2 * 16(%%rax), %%xmm0  \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "movaps  -1 * 16(%%rax), %%xmm1  \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t" // iteration 3
        "movaps  -1 * 16(%%rbx), %%xmm3  \n\t"
        "addpd   %%xmm4, %%xmm15         \n\t"
        "movaps  %%xmm2, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
        "mulpd   %%xmm0, %%xmm2          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "subq  $-4 * 4 * 8, %%rax        \n\t" // a += 4*4 (unroll x mr)
        "                                \n\t"
        "addpd   %%xmm5, %%xmm10         \n\t"
        "addpd   %%xmm6, %%xmm14         \n\t"
        "movaps  %%xmm7, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm7          \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "                                \n\t"
        "subq  $-4 * 4 * 8, %%r9         \n\t" // b_next += 4*4 (unroll x nr)
        "                                \n\t"
        "addpd   %%xmm2, %%xmm9          \n\t"
        "movaps   0 * 16(%%rbx), %%xmm2  \n\t"
        "addpd   %%xmm4, %%xmm13         \n\t"
        "movaps  %%xmm3, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
        "mulpd   %%xmm0, %%xmm3          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "subq  $-4 * 4 * 8, %%rbx        \n\t" // b += 4*4 (unroll x nr)
        "                                \n\t"
        "addpd   %%xmm7, %%xmm8          \n\t"
        "addpd   %%xmm6, %%xmm12         \n\t"
        "movaps  %%xmm5, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm5          \n\t"
        "movaps  -8 * 16(%%rax), %%xmm0  \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "movaps  -7 * 16(%%rax), %%xmm1  \n\t"
        "                                \n\t"
        "prefetcht2        0 * 8(%%r9)   \n\t" // prefetch b_next[0]
        "prefetcht2        8 * 8(%%r9)   \n\t" // prefetch b_next[8]
        "                                \n\t"
        "decq   %%rsi                    \n\t" // i -= 1;
        "jne    .DLOOPKITER              \n\t" // iterate again if i != 0.
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        //"prefetcht2       -8 * 8(%%r9)   \n\t" // prefetch b_next[-8]
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        ".DCONSIDKLEFT:                  \n\t"
        "                                \n\t"
        "movq      %1, %%rsi             \n\t" // i = k_left;
        "testq  %%rsi, %%rsi             \n\t" // check i via logical AND.
        "je     .DPOSTACCUM              \n\t" // if i == 0, we're done; jump to end.
        "                                \n\t" // else, we prepare to enter k_left loop.
        "                                \n\t"
        "                                \n\t"
        ".DLOOPKLEFT:                    \n\t" // EDGE LOOP
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t" // iteration 0
        "movaps  -7 * 16(%%rbx), %%xmm3  \n\t"
        "addpd   %%xmm4, %%xmm15         \n\t"
        "movaps  %%xmm2, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
        "mulpd   %%xmm0, %%xmm2          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "addpd   %%xmm5, %%xmm10         \n\t"
        "addpd   %%xmm6, %%xmm14         \n\t"
        "movaps  %%xmm7, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm7          \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "                                \n\t"
        "addpd   %%xmm2, %%xmm9          \n\t"
        "movaps  -6 * 16(%%rbx), %%xmm2  \n\t"
        "addpd   %%xmm4, %%xmm13         \n\t"
        "movaps  %%xmm3, %%xmm4          \n\t"
        "pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
        "mulpd   %%xmm0, %%xmm3          \n\t"
        "mulpd   %%xmm1, %%xmm4          \n\t"
        "                                \n\t"
        "addpd   %%xmm7, %%xmm8          \n\t"
        "addpd   %%xmm6, %%xmm12         \n\t"
        "movaps  %%xmm5, %%xmm6          \n\t"
        "mulpd   %%xmm0, %%xmm5          \n\t"
        "movaps  -6 * 16(%%rax), %%xmm0  \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "movaps  -5 * 16(%%rax), %%xmm1  \n\t"
        "                                \n\t"
        "                                \n\t"
        "subq  $-4 * 1 * 8, %%rax        \n\t" // a += 4 (1 x mr)
        "subq  $-4 * 1 * 8, %%rbx        \n\t" // b += 4 (1 x nr)
        "                                \n\t"
        "                                \n\t"
        "decq   %%rsi                    \n\t" // i -= 1;
        "jne    .DLOOPKLEFT              \n\t" // iterate again if i != 0.
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        ".DPOSTACCUM:                    \n\t"
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t"
        "addpd   %%xmm4, %%xmm15         \n\t"
        "addpd   %%xmm5, %%xmm10         \n\t"
        "addpd   %%xmm6, %%xmm14         \n\t"
        "                                \n\t"
        "                                \n\t"
        "movsd   %4, %%xmm6              \n\t" // load alpha
        "movsd   %5, %%xmm7              \n\t" // load beta 
        "unpcklpd %%xmm6, %%xmm6         \n\t" // duplicate alpha
        "unpcklpd %%xmm7, %%xmm7         \n\t" // duplicate beta
        "                                \n\t"
        "                                \n\t"
        "movq    %7, %%rsi               \n\t" // load rs_c
        "movq    %%rsi, %%r8             \n\t" // make a copy of rs_c
        "                                \n\t"
        "leaq    (,%%rsi,8), %%rsi       \n\t" // rsi = rs_c * sizeof(double)
        "                                \n\t"
        "leaq   (%%rcx,%%rsi,2), %%rdx   \n\t" // load address of c + 2*rs_c;
        "                                \n\t"
        "                                \n\t" // xmm8:   xmm9:   xmm10:  xmm11:
        "                                \n\t" // ( ab01  ( ab00  ( ab03  ( ab02
        "                                \n\t" //   ab10 )  ab11 )  ab12 )  ab13 )
        "                                \n\t" //
        "                                \n\t" // xmm12:  xmm13:  xmm14:  xmm15:
        "                                \n\t" // ( ab21  ( ab20  ( ab23  ( ab22
        "                                \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
        "movaps   %%xmm8,  %%xmm0        \n\t"
        "movsd    %%xmm9,  %%xmm8        \n\t"
        "movsd    %%xmm0,  %%xmm9        \n\t"
        "                                \n\t"
        "movaps  %%xmm10,  %%xmm0        \n\t"
        "movsd   %%xmm11, %%xmm10        \n\t"
        "movsd    %%xmm0, %%xmm11        \n\t"
        "                                \n\t"
        "movaps  %%xmm12,  %%xmm0        \n\t"
        "movsd   %%xmm13, %%xmm12        \n\t"
        "movsd    %%xmm0, %%xmm13        \n\t"
        "                                \n\t"
        "movaps  %%xmm14,  %%xmm0        \n\t"
        "movsd   %%xmm15, %%xmm14        \n\t"
        "movsd    %%xmm0, %%xmm15        \n\t"
        "                                \n\t" // xmm8:   xmm9:   xmm10:  xmm11:
        "                                \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
        "                                \n\t" //   ab10 )  ab11 )  ab12 )  ab13 )
        "                                \n\t" //
        "                                \n\t" // xmm12:  xmm13:  xmm14:  xmm15:
        "                                \n\t" // ( ab20  ( ab21  ( ab22  ( ab23
        "                                \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t" // determine if
        "                                \n\t" //   c % 16 == 0, AND
        "                                \n\t" //   rs_c == 1
        "                                \n\t" // ie: aligned and column-stored
        "                                \n\t"
        "cmpq       $1, %%r8             \n\t" // set ZF if rs_c == 1.
        "sete           %%bl             \n\t" // bl = ( ZF == 1 ? 1 : 0 );
        "testq     $15, %%rcx            \n\t" // set ZF if c & 16 is zero.
        "setz           %%bh             \n\t" // bh = ( ZF == 1 ? 1 : 0 );
        "                                \n\t" // and(bl,bh) will reveal result
        "                                \n\t"
        "                                \n\t" // now avoid loading C if beta == 0
        "                                \n\t"
        "xorpd     %%xmm0,  %%xmm0       \n\t" // set xmm0 to zero.
        "ucomisd   %%xmm0,  %%xmm7       \n\t" // check if beta == 0.
        "je      .DBETAZERO              \n\t" // if ZF = 1, jump to beta == 0 case
        "                                \n\t"
        "                                \n\t"
        "                                \n\t" // check if aligned/column-stored
        "andb     %%bl, %%bh             \n\t" // set ZF if bl & bh == 1.
        "jne     .DCOLSTORED             \n\t" // jump to column storage case
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        ".DGENSTORED:                    \n\t"
        "                                \n\t"
        "movlpd  (%%rcx),       %%xmm0   \n\t" // load c00 and c10,
        "movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
        "mulpd   %%xmm6,  %%xmm8         \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
        "addpd   %%xmm8,  %%xmm0         \n\t" // add the gemm result,
        "movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
        "movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t"
        "movlpd  (%%rdx),       %%xmm1   \n\t" // load c20 and c30,
        "movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
        "mulpd   %%xmm6,  %%xmm12        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
        "addpd  %%xmm12,  %%xmm1         \n\t" // add the gemm result,
        "movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
        "movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "movlpd  (%%rcx),       %%xmm0   \n\t" // load c01 and c11,
        "movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
        "mulpd   %%xmm6,  %%xmm9         \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
        "addpd   %%xmm9,  %%xmm0         \n\t" // add the gemm result,
        "movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
        "movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t"
        "movlpd  (%%rdx),       %%xmm1   \n\t" // load c21 and c31,
        "movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
        "mulpd   %%xmm6,  %%xmm13        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
        "addpd  %%xmm13,  %%xmm1         \n\t" // add the gemm result,
        "movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
        "movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "movlpd  (%%rcx),       %%xmm0   \n\t" // load c02 and c12,
        "movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
        "mulpd   %%xmm6,  %%xmm10        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
        "addpd  %%xmm10,  %%xmm0         \n\t" // add the gemm result,
        "movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
        "movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t"
        "movlpd  (%%rdx),       %%xmm1   \n\t" // load c22 and c32,
        "movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
        "mulpd   %%xmm6,  %%xmm14        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
        "addpd  %%xmm14,  %%xmm1         \n\t" // add the gemm result,
        "movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
        "movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "movlpd  (%%rcx),       %%xmm0   \n\t" // load c03 and c13,
        "movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
        "mulpd   %%xmm6,  %%xmm11        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
        "addpd  %%xmm11,  %%xmm0         \n\t" // add the gemm result,
        "movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
        "movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
        "                                \n\t"
        "                                \n\t"
        "movlpd  (%%rdx),       %%xmm1   \n\t" // load c23 and c33,
        "movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
        "mulpd   %%xmm6,  %%xmm15        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
        "addpd  %%xmm15,  %%xmm1         \n\t" // add the gemm result,
        "movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
        "movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
        "                                \n\t"
        "jmp    .DDONE                   \n\t" // jump to end.
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        ".DCOLSTORED:                    \n\t"
        "                                \n\t"
        "movaps  (%%rcx),       %%xmm0   \n\t" // load c00 and c10,
        "mulpd   %%xmm6,  %%xmm8         \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
        "addpd   %%xmm8,  %%xmm0         \n\t" // add the gemm result,
        "movaps  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t"
        "movaps  (%%rdx),       %%xmm1   \n\t" // load c20 and c30,
        "mulpd   %%xmm6,  %%xmm12        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
        "addpd  %%xmm12,  %%xmm1         \n\t" // add the gemm result,
        "movaps  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "movaps  (%%rcx),       %%xmm0   \n\t" // load c01 and c11,
        "mulpd   %%xmm6,  %%xmm9         \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
        "addpd   %%xmm9,  %%xmm0         \n\t" // add the gemm result,
        "movaps  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t"
        "movaps  (%%rdx),       %%xmm1   \n\t" // load c21 and c31,
        "mulpd   %%xmm6,  %%xmm13        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
        "addpd  %%xmm13,  %%xmm1         \n\t" // add the gemm result,
        "movaps  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "movaps  (%%rcx),       %%xmm0   \n\t" // load c02 and c12,
        "mulpd   %%xmm6,  %%xmm10        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
        "addpd  %%xmm10,  %%xmm0         \n\t" // add the gemm result,
        "movaps  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t"
        "movaps  (%%rdx),       %%xmm1   \n\t" // load c22 and c32,
        "mulpd   %%xmm6,  %%xmm14        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
        "addpd  %%xmm14,  %%xmm1         \n\t" // add the gemm result,
        "movaps  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "movaps  (%%rcx),       %%xmm0   \n\t" // load c03 and c13,
        "mulpd   %%xmm6,  %%xmm11        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
        "addpd  %%xmm11,  %%xmm0         \n\t" // add the gemm result,
        "movaps  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
        "                                \n\t"
        "                                \n\t"
        "movaps  (%%rdx),       %%xmm1   \n\t" // load c23 and c33,
        "mulpd   %%xmm6,  %%xmm15        \n\t" // scale by alpha,
        "mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
        "addpd  %%xmm15,  %%xmm1         \n\t" // add the gemm result,
        "movaps  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
        "                                \n\t"
        "jmp    .DDONE                   \n\t" // jump to end.
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        ".DBETAZERO:                     \n\t"
        "                                \n\t" // check if aligned/column-stored
        "andb     %%bl, %%bh             \n\t" // set ZF if bl & bh == 1.
        "jne     .DCOLSTORBZ             \n\t" // jump to column storage case
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        ".DGENSTORBZ:                    \n\t"
        "                                \n\t" // skip loading c00 and c10,
        "mulpd   %%xmm6,  %%xmm8         \n\t" // scale by alpha,
        "movlpd  %%xmm8,  (%%rcx)        \n\t" // and store back to memory.
        "movhpd  %%xmm8,  (%%rcx,%%rsi)  \n\t"
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t" // skip loading c20 and c30,
        "mulpd   %%xmm6,  %%xmm12        \n\t" // scale by alpha,
        "movlpd  %%xmm12, (%%rdx)        \n\t" // and store back to memory.
        "movhpd  %%xmm12, (%%rdx,%%rsi)  \n\t"
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t" // skip loading c01 and c11,
        "mulpd   %%xmm6,  %%xmm9         \n\t" // scale by alpha,
        "movlpd  %%xmm9,  (%%rcx)        \n\t" // and store back to memory.
        "movhpd  %%xmm9,  (%%rcx,%%rsi)  \n\t"
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t" // skip loading c21 and c31,
        "mulpd   %%xmm6,  %%xmm13        \n\t" // scale by alpha,
        "movlpd  %%xmm13, (%%rdx)        \n\t" // and store back to memory.
        "movhpd  %%xmm13, (%%rdx,%%rsi)  \n\t"
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t" // skip loading c02 and c12,
        "mulpd   %%xmm6,  %%xmm10        \n\t" // scale by alpha,
        "movlpd  %%xmm10, (%%rcx)        \n\t" // and store back to memory.
        "movhpd  %%xmm10, (%%rcx,%%rsi)  \n\t"
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t" // skip loading c22 and c32,
        "mulpd   %%xmm6,  %%xmm14        \n\t" // scale by alpha,
        "movlpd  %%xmm14, (%%rdx)        \n\t" // and store back to memory.
        "movhpd  %%xmm14, (%%rdx,%%rsi)  \n\t"
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t" // skip loading c03 and c13,
        "mulpd   %%xmm6,  %%xmm11        \n\t" // scale by alpha,
        "movlpd  %%xmm11, (%%rcx)        \n\t" // and store back to memory.
        "movhpd  %%xmm11, (%%rcx,%%rsi)  \n\t"
        "                                \n\t"
        "                                \n\t" // skip loading c23 and c33,
        "mulpd   %%xmm6,  %%xmm15        \n\t" // scale by alpha,
        "movlpd  %%xmm15, (%%rdx)        \n\t" // and store back to memory.
        "movhpd  %%xmm15, (%%rdx,%%rsi)  \n\t"
        "                                \n\t"
        "jmp    .DDONE                   \n\t" // jump to end.
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        ".DCOLSTORBZ:                    \n\t"
        "                                \n\t"
        "                                \n\t" // skip loading c00 and c10,
        "mulpd   %%xmm6,  %%xmm8         \n\t" // scale by alpha,
        "movaps  %%xmm8,  (%%rcx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t" // skip loading c20 and c30,
        "mulpd   %%xmm6,  %%xmm12        \n\t" // scale by alpha,
        "movaps  %%xmm12, (%%rdx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t" // skip loading c01 and c11,
        "mulpd   %%xmm6,  %%xmm9         \n\t" // scale by alpha,
        "movaps  %%xmm9,  (%%rcx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t" // skip loading c21 and c31,
        "mulpd   %%xmm6,  %%xmm13        \n\t" // scale by alpha,
        "movaps  %%xmm13, (%%rdx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t" // skip loading c02 and c12,
        "mulpd   %%xmm6,  %%xmm10        \n\t" // scale by alpha,
        "movaps  %%xmm10, (%%rcx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rcx           \n\t"
        "                                \n\t" // skip loading c22 and c32,
        "mulpd   %%xmm6,  %%xmm14        \n\t" // scale by alpha,
        "movaps  %%xmm14, (%%rdx)        \n\t" // and store back to memory.
        "addq     %%rdi, %%rdx           \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t" // skip loading c03 and c13,
        "mulpd   %%xmm6,  %%xmm11        \n\t" // scale by alpha,
        "movaps  %%xmm11, (%%rcx)        \n\t" // and store back to memory.
        "                                \n\t"
        "                                \n\t" // skip loading c23 and c33,
        "mulpd   %%xmm6,  %%xmm15        \n\t" // scale by alpha,
        "movaps  %%xmm15, (%%rdx)        \n\t" // and store back to memory.
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        ".DDONE:                         \n\t"
        "                                \n\t"

        : // output operands (none)
        : // input operands
          "m" (kb),     // 0
          "m" (kl),     // 1
          "m" (A),      // 2
          "m" (B),      // 3
          "m" (alpha),  // 4
          "m" (beta),   // 5
          "m" (C),      // 6
          "m" (incRowC),   // 7
          "m" (incColC),   // 8
          "m" (nextB), // 9
          "m" (nextA)  // 10
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11",
          "xmm0", "xmm1", "xmm2", "xmm3",
          "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11",
          "xmm12", "xmm13", "xmm14", "xmm15",
          "memory"
    );
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
