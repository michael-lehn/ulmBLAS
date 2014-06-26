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
double _A[MC*KC];
double _B[KC*NC];
double _C[MR*NR];

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
dgemm_micro_kernel(int kc,
                   double alpha, const double *A, const double *B,
                   double beta,
                   double *C, int incRowC, int incColC,
                   const double *nextA, const double *nextB)
{
    double AB[4*4] __attribute__ ((aligned (16)));

    __m128d  a_0_1, a_2_3, b_0_1, b_1_0;
    __m128d  d_00_11, d_01_10;
    __m128d  d_20_31, d_21_30;

    __m128d  ab_00_11, ab_01_10, ab_02_13, ab_03_12;
    __m128d  ab_20_31, ab_21_30, ab_22_33, ab_23_32;

    int  i, j, l;

    ab_00_11 = _mm_setzero_pd();
    ab_01_10 = _mm_setzero_pd();
    ab_02_13 = _mm_setzero_pd();
    ab_03_12 = _mm_setzero_pd();

    ab_20_31 = _mm_setzero_pd();
    ab_21_30 = _mm_setzero_pd();
    ab_22_33 = _mm_setzero_pd();
    ab_23_32 = _mm_setzero_pd();


    for (l=0; l<kc; ++l) {
        // compute diag pairs (a[0]*b[0],a[1]*b[1]) and (a[2]*b[0],a[3]*b[1])
        a_0_1 = _mm_load_pd(A);
        a_2_3 = _mm_load_pd(A+2);

        b_0_1 = _mm_load_pd(B);

        d_00_11 = a_0_1;
        d_20_31 = a_2_3;

        d_00_11 = _mm_mul_pd(d_00_11, b_0_1);
        d_20_31 = _mm_mul_pd(d_20_31, b_0_1);

        ab_00_11 = _mm_add_pd(ab_00_11, d_00_11);
        ab_20_31 = _mm_add_pd(ab_20_31, d_20_31);


        // compute diag pairs (a[0]*b[1],a[1]*b[0]) and (a[2]*b[1],a[3]*b[0])
        b_1_0 = _mm_shuffle_pd(b_0_1, b_0_1, _MM_SHUFFLE2(0, 1));

        d_01_10 = a_0_1;
        d_21_30 = a_2_3;

        d_01_10 = _mm_mul_pd(d_01_10, b_1_0);
        d_21_30 = _mm_mul_pd(d_21_30, b_1_0);

        ab_01_10 = _mm_add_pd(ab_01_10, d_01_10);
        ab_21_30 = _mm_add_pd(ab_21_30, d_21_30);

        // compute diag pairs (a[0]*b[2],a[1]*b[3]) and (a[2]*b[2],a[3]*b[3])
        b_0_1 = _mm_load_pd(B+2);

        d_00_11 = a_0_1;
        d_20_31 = a_2_3;

        d_00_11 = _mm_mul_pd(d_00_11, b_0_1);
        d_20_31 = _mm_mul_pd(d_20_31, b_0_1);

        ab_02_13 = _mm_add_pd(ab_02_13, d_00_11);
        ab_22_33 = _mm_add_pd(ab_22_33, d_20_31);

        // compute diag pairs (a[0]*b[3],a[1]*b[2]) and (a[2]*b[3],a[3]*b[2])
        b_1_0 = _mm_shuffle_pd(b_0_1, b_0_1, _MM_SHUFFLE2(0, 1));

        d_01_10 = a_0_1;
        d_21_30 = a_2_3;

        d_01_10 = _mm_mul_pd(d_01_10, b_1_0);
        d_21_30 = _mm_mul_pd(d_21_30, b_1_0);

        ab_03_12 = _mm_add_pd(ab_03_12, d_01_10);
        ab_23_32 = _mm_add_pd(ab_23_32, d_21_30);

        // move on
        A += 4;
        B += 4;
    }

    _mm_storel_pd(&AB[0+0*4], ab_00_11);
    _mm_storeh_pd(&AB[1+0*4], ab_01_10);
    _mm_storel_pd(&AB[2+0*4], ab_20_31);
    _mm_storeh_pd(&AB[3+0*4], ab_21_30);

    _mm_storel_pd(&AB[0+1*4], ab_01_10);
    _mm_storeh_pd(&AB[1+1*4], ab_00_11);
    _mm_storel_pd(&AB[2+1*4], ab_21_30);
    _mm_storeh_pd(&AB[3+1*4], ab_20_31);

    _mm_storel_pd(&AB[0+2*4], ab_02_13);
    _mm_storeh_pd(&AB[1+2*4], ab_03_12);
    _mm_storel_pd(&AB[2+2*4], ab_22_33);
    _mm_storeh_pd(&AB[3+2*4], ab_23_32);

    _mm_storel_pd(&AB[0+3*4], ab_03_12);
    _mm_storeh_pd(&AB[1+3*4], ab_02_13);
    _mm_storel_pd(&AB[2+3*4], ab_23_32);
    _mm_storeh_pd(&AB[3+3*4], ab_22_33);

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
