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

#if MR==4 && NR==4
//
//  Micro kernel for multiplying panels from A and B.
//
static void
dgemm_micro_kernel(int kc,
                   double alpha, const double *A, const double *B,
                   double beta,
                   double *C, int incRowC, int incColC)
{
    double _AB[MR*NR] __attribute__ ((aligned (16)));
    double *AB = _AB;

    int i, j;

//
//  Compute AB = A*B
//
    __asm__ volatile
    (
        "movq        %1, %%rax           \n\t"
        "movq        %2, %%rbx           \n\t"
        "movq        %3, %%rcx           \n\t"
        "                                \n\t"
        "xorpd   %%xmm3, %%xmm3          \n\t"
        "xorpd   %%xmm4, %%xmm4          \n\t"
        "xorpd   %%xmm5, %%xmm5          \n\t"
        "xorpd   %%xmm6, %%xmm6          \n\t"
        "                                \n\t"
        "xorpd   %%xmm8, %%xmm8          \n\t"
        "xorpd   %%xmm9, %%xmm9          \n\t"
        "xorpd   %%xmm10, %%xmm10        \n\t"
        "xorpd   %%xmm11, %%xmm11        \n\t"
        "xorpd   %%xmm12, %%xmm12        \n\t"
        "xorpd   %%xmm13, %%xmm13        \n\t"
        "xorpd   %%xmm14, %%xmm14        \n\t"
        "xorpd   %%xmm15, %%xmm15        \n\t"
        "                                \n\t"
        "movaps    (%%rax), %%xmm0       \n\t"
        "movaps  16(%%rax), %%xmm1       \n\t"
        "movaps    (%%rbx), %%xmm2       \n\t"
        "                                \n\t"
        "movl        %0, %%esi           \n\t"
        "testl    %%esi, %%esi           \n\t"
        "je      .DWRITEBACK%=           \n\t"
        "                                \n\t"
        ".DLOOP%=:                       \n\t"
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t"
        "movaps  16(%%rbx), %%xmm3       \n\t"
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
        "movaps  32(%%rbx), %%xmm2       \n\t"
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
        "movaps  32(%%rax), %%xmm0       \n\t"
        "mulpd   %%xmm1, %%xmm6          \n\t"
        "movaps  48(%%rax), %%xmm1       \n\t"
        "                                \n\t"
        "                                \n\t"
        "addq    $32, %%rax              \n\t"
        "addq    $32, %%rbx              \n\t"
        "                                \n\t"
        "                                \n\t"
        "decl   %%esi                    \n\t"
        "jne    .DLOOP%=                 \n\t"
        "                                \n\t"
        "                                \n\t"
        "addpd   %%xmm3, %%xmm11         \n\t"
        "addpd   %%xmm4, %%xmm15         \n\t"
        "addpd   %%xmm5, %%xmm10         \n\t"
        "addpd   %%xmm6, %%xmm14         \n\t"
        "                                \n\t"
        ".DWRITEBACK%=:                  \n\t"
        "                                \n\t"
        "movlpd  %%xmm9,    (%%rcx)      \n\t"
        "movhpd  %%xmm8,   8(%%rcx)      \n\t"
        "movlpd  %%xmm13, 16(%%rcx)      \n\t"
        "movhpd  %%xmm12, 24(%%rcx)      \n\t"
        "                                \n\t"
        "addq  $32, %%rcx                \n\t"
        "movlpd  %%xmm8,    (%%rcx)      \n\t"
        "movhpd  %%xmm9,   8(%%rcx)      \n\t"
        "movlpd  %%xmm12, 16(%%rcx)      \n\t"
        "movhpd  %%xmm13, 24(%%rcx)      \n\t"
        "                                \n\t"
        "addq  $32, %%rcx                \n\t"
        "movlpd  %%xmm11,   (%%rcx)      \n\t"
        "movhpd  %%xmm10,  8(%%rcx)      \n\t"
        "movlpd  %%xmm15, 16(%%rcx)      \n\t"
        "movhpd  %%xmm14, 24(%%rcx)      \n\t"
        "                                \n\t"
        "addq  $32, %%rcx                \n\t"
        "movlpd  %%xmm10,   (%%rcx)      \n\t"
        "movhpd  %%xmm11,  8(%%rcx)      \n\t"
        "movlpd  %%xmm14, 16(%%rcx)      \n\t"
        "movhpd  %%xmm15, 24(%%rcx)      \n\t"
    : // output
    : // input
        "m" (kc),     // 0
        "m" (A),      // 1
        "m" (B),      // 2
        "m" (AB)      // 3
    : // register clobber list
        "rax", "rbx", "rcx", "esi",
        "xmm0", "xmm1", "xmm2", "xmm3",
        "xmm4", "xmm5", "xmm6", "xmm7",
        "xmm8", "xmm9", "xmm10", "xmm11",
        "xmm12", "xmm13", "xmm14", "xmm15"
    );

//
//  Update C <- beta*C
//
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
//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
//                                  the above layer dgemm_nn)
//
    if (alpha==1.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*MR];
            }
        }
    } else {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += alpha*AB[i+j*MR];
            }
        }
    }
}
#else
#   error "This micro kernel requires MR==4 and NR==4"
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

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

            if (mr==MR && nr==NR) {
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                   beta,
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC);
            } else {
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                   0.0,
                                   _C, 1, MR);
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
