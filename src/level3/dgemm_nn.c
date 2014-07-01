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
static double _A[MC*KC] __attribute__ ((aligned (16)));
static double _B[KC*NC] __attribute__ ((aligned (16)));
static double _C[MR*NR] __attribute__ ((aligned (16)));

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
    double _AB[4*4] __attribute__ ((aligned (16)));
    double *AB = _AB;

    int i, j;

    __asm__ volatile
    (
        "                                \n\t"
        "                                \n\t"
        "movq          %1, %%rax         \n\t" // load address of a.
        "movq          %2, %%rbx         \n\t" // load address of b.
        "                                \n\t"
        "                                \n\t"
        "movaps         (%%rax), %%xmm0  \n\t" // initialize loop by pre-loading elements
        "movaps       16(%%rax), %%xmm1  \n\t" // of a and b.
        "movaps         (%%rbx), %%xmm2  \n\t"
        "                                \n\t"
        "                                \n\t"
        "xorpd     %%xmm3,  %%xmm3       \n\t"
        "xorpd     %%xmm4,  %%xmm4       \n\t"
        "xorpd     %%xmm5,  %%xmm5       \n\t"
        "xorpd     %%xmm6,  %%xmm6       \n\t"
        "                                \n\t"
        "xorpd     %%xmm8,  %%xmm8       \n\t"
        "movaps    %%xmm8,  %%xmm9       \n\t"
        "movaps    %%xmm8, %%xmm10       \n\t"
        "movaps    %%xmm8, %%xmm11       \n\t"
        "movaps    %%xmm8, %%xmm12       \n\t"
        "movaps    %%xmm8, %%xmm13       \n\t"
        "movaps    %%xmm8, %%xmm14       \n\t"
        "movaps    %%xmm8, %%xmm15       \n\t"
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        "movq      %0, %%rsi             \n\t" // i = k_left;
        "testq  %%rsi, %%rsi             \n\t" // check i via logical AND.
        "je     .DPOSTACCUM%=            \n\t" // if i == 0, we're done; jump to end.
        "                                \n\t" // else, we prepare to enter k_left loop.
        "                                \n\t"
        "                                \n\t"
        ".DLOOPKLEFT%=:                  \n\t" // EDGE LOOP
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t" // iteration 0
        "movaps       16(%%rbx), %%xmm3  \n\t"
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
        "movaps       32(%%rbx), %%xmm2  \n\t"
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
        "movaps       32(%%rax), %%xmm0  \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "movaps       48(%%rax), %%xmm1  \n\t"
        "                                \n\t"
        "                                \n\t"
        "addq   $4 * 1 * 8, %%rax        \n\t" // a += 4 (1 x mr)
        "addq   $4 * 1 * 8, %%rbx        \n\t" // b += 4 (1 x nr)
        "                                \n\t"
        "                                \n\t"
        "decq   %%rsi                    \n\t" // i -= 1;
        "jne    .DLOOPKLEFT%=            \n\t" // iterate again if i != 0.
        "                                \n\t"
        "                                \n\t"
        "                                \n\t"
        ".DPOSTACCUM%=:                  \n\t"
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t"
        "addpd   %%xmm4, %%xmm15         \n\t"
        "addpd   %%xmm5, %%xmm10         \n\t"
        "addpd   %%xmm6, %%xmm14         \n\t"
        "                                \n\t"
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
        ".DDONE%=:                       \n\t"
        "movq          %3,      %%r8     \n\t" // load address of AB.
        "movaps   %%xmm8,      (%%r8)    \n\t"
        "movaps   %%xmm12,   16(%%r8)    \n\t"
        "                                \n\t"
        "movaps   %%xmm9,    32(%%r8)    \n\t"
        "movaps   %%xmm13,   48(%%r8)    \n\t"
        "                                \n\t"
        "movaps   %%xmm10,   64(%%r8)    \n\t"
        "movaps   %%xmm14,   80(%%r8)    \n\t"
        "                                \n\t"
        "movaps   %%xmm11,   96(%%r8)    \n\t"
        "movaps   %%xmm15,  112(%%r8)    \n\t"
        "                                \n\t"

        : // output operands (none)
        : // input operands
          "m" (kc),     // 0
          "m" (A),      // 1
          "m" (B),      // 2
          "m" (AB)      // 3
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11",
          "xmm0", "xmm1", "xmm2", "xmm3",
          "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11",
          "xmm12", "xmm13", "xmm14", "xmm15",
          "memory"
    );

    if (beta==0.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

//
//  Update C <- C + alpha*A*B
//
    if (alpha==1.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*NR];
            }
        }
    } else if (alpha!=0.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += alpha*AB[i+j*NR];
            }
        }
    }
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
