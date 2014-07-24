#include <ulmblas.h>
#include <level3/dgemm_nn.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

//
//  Buffer size of _A results from partitioning a lower triangular MC x MC
//  matrix into MR x MR blocks.
//
static double _A[MR*MR*(MC/MR)*(MC/MR+1)/2];
static double _B[MC*NC];

//
//  Pack the lower triangular mr x mr block from A into buffer.  Padding gets
//  used such that the buffered block represents a MR x MR block with MR >= mr.
//  Padding inserts ones on the diagonal and zeros otherwise.  For diagonal
//  elements its reciprocal value gets stored.
//
static void
pack_A_trblock_MRxMR(int unitDiag, int mr,
                     const double *A, int incRowA, int incColA,
                     double *buffer)
{
    int i, j;

    assert(mr<=MR);

    for (j=0; j<MR; ++j) {
        for (i=0; i<j; ++i) {
            buffer[i] = 0.0;
        }
        if (j<mr) {
            buffer[j] = (unitDiag) ? 1.0
                                   : 1.0/A[j*incRowA+j*incColA];
        } else {
            buffer[j] = 1.0;
        }
        for (i=j+1; i<MR; ++i) {
            if (i<mr && j<mr) {
                buffer[i] = A[i*incRowA+j*incColA];
            } else if (i!=j) {
                buffer[i] = 0.0;
            } else {
                buffer[i] = 1.0;
            }
        }
        buffer += MR;
    }
}

//
//  Pack a rectangular mr x MR block from A into buffer.  Using zero padding the
//  block gets extended to size MR x MR
//
static void
pack_A_geblock_mrxMR(int mr, const double *A, int incRowA, int incColA,
                     double *buffer)
{
    int i, j;

    assert(mr<=MR);

    for (j=0; j<MR; ++j) {
        for (i=0; i<MR; ++i) {
            if (i<mr) {
                buffer[i] = A[i*incRowA+j*incColA];
            } else {
                buffer[i] = 0.0;
            }
        }
        buffer += MR;
    }
}

//
//  Partition the lower triangular mc x mc matrix A into MR x MR blocks and pack
//  these column wise into the buffer.  Padding is used for smaller blocks at
//  right or bottom boundary.
//
static void
pack_A(int unitDiag, int mc, const double *A, int incRowA, int incColA,
       double *buffer)
{
    int mp  = (mc+MR-1) / MR;
    int _mr = mc % MR;

    int i, j;

    for (j=0; j<mp; ++j) {
        int mr = (j!=mp-1 || _mr==0) ? MR : _mr;

        pack_A_trblock_MRxMR(unitDiag, mr,
                             &A[j*MR*incRowA+j*MR*incColA], incRowA, incColA,
                             buffer);
        buffer += MR*MR;

        for (i=j+1; i<mp; ++i) {
            mr = (i!=mp-1 || _mr==0) ? MR : _mr;

            pack_A_geblock_mrxMR(mr,
                                 &A[i*MR*incRowA
                                   +j*MR*incColA], incRowA, incColA,
                                 buffer);
            buffer += MR*MR;
        }
    }
}

//
//  Partition the rectangular mc x nc matrix B in vertical panels of width NR.
//  Zero padding is used for the last panel.  Zero padding is also used such
//  that all panels have a height that is a multiple of MR.
//
static void
pack_B(int mc, int nc, const double *B, int incRowB, int incColB,
       double *buffer)
{
    int  mp  = (mc+MR-1) / MR;
    int  np  = (nc+NR-1) / NR;
    int _nr  = nc % NR;

    int i, j, l;

    for (j=0; j<np; ++j) {
        int nr = (j!=np-1 || _nr==0) ? NR : _nr;

        for (i=0; i<mc; ++i) {
            for (l=0; l<nr; ++l) {
                buffer[l] = B[i*incRowB+l*incColB];
            }
            for (l=nr; l<NR; ++l) {
                buffer[l] = 0.0;
            }
            buffer += NR;
        }
        for (i=mc; i<MR*mp; ++i) {
            for (l=0; l<NR; ++l) {
                buffer[l] = 0.0;
            }
            buffer += NR;
        }

        B += NR*incColB;
    }
}

static void
unpack_B(const double *buffer,
         int mc, int nc, double *B, int incRowB, int incColB)
{
    int  mp  = (mc+MR-1) / MR;
    int  np  = (nc+NR-1) / NR;
    int _nr  = nc % NR;

    int i, j, l;

    for (j=0; j<np; ++j) {
        int nr = (j!=np-1 || _nr==0) ? NR : _nr;

        for (i=0; i<mc; ++i) {
            for (l=0; l<nr; ++l) {
                B[i*incRowB+l*incColB] = buffer[l];
            }
            buffer += NR;
        }

        buffer += NR*(MR*mp-mc);
        B      += NR*incColB;
    }
}

static void
dtrsm_l_micro_kernel(double alpha, const double *A, double *B)
{
    int i, j, k;

    if (alpha!=1.0) {
        for (i=0; i<MR; ++i) {
            for (j=0; j<NR; ++j) {
                B[i*NR+j] *= alpha;
            }
        }
    }
    for (j=0; j<NR; ++j) {
        for (k=0; k<MR; ++k) {
            B[k*NR+j] *= A[k+k*MR];
            for (i=k+1; i<MR; ++i) {
                B[i*NR+j] -= B[k*NR+j]*A[i+k*MR];
            }
        }
    }
}

static void
dtrsm_l_macro_kernel(int mc, int nc, double alpha)
{
    int mp = (mc+MR-1)/MR;
    int np = (nc+NR-1)/NR;

    int i, j, k;

    double _alpha;

    for (k=0; k<np; ++k) {
        for (j=0; j<mp; ++j) {
            _alpha = (j==0) ? alpha : 1.0;

            dtrsm_l_micro_kernel(_alpha,
                                 &_A[(j+j*(2*mp-j-1)/2)*MR*MR],
                                 &_B[(j+k*mp)*MR*NR]);

            for (i=j+1; i<mp; ++i) {
                dgemm_micro_kernel(MR, -1.0,
                                   &_A[(i+j*(2*mp-j-1)/2)*MR*MR],
                                   &_B[(j+k*mp)*MR*NR],
                                   alpha,
                                   &_B[(i+k*mp)*MR*NR], NR, 1,
                                   0, 0);
            }
        }
    }
}

static void
printMatrix(int m, int n, const double *X, int incRowX, int incColX)
{
    int i, j;

    for (i=0; i<m; ++i) {
        for (j=0; j<n; ++j) {
            printf("  %9.5lf", X[i*incRowX+j*incColX]);
        }
        printf("\n");
    }
    printf("\n");
}

void
dtrsm_l(int             unitDiag,
        int             m,
        int             n,
        double          alpha,
        const double    *A,
        int             incRowA,
        int             incColA,
        double          *B,
        int             incRowB,
        int             incColB)
{
    int mb = (m+MC-1)/MC;
    int nb = (n+NC-1)/NC;

    int _mc = m % MC;
    int _nc = n % NC;

    int mc, nc;
    int j, k;

    double _alpha;

    /*
    printf("A =\n");
    printMatrix(m, m, A, incRowA, incColA);

    printf("B =\n");
    printMatrix(m, n, B, incRowB, incColB);

    printf("unitDiag = %d\n", unitDiag);
    */

    if (alpha!=1.0) {
        for (j=0; j<n; ++j) {
            for (k=0; k<m; ++k) {
                B[k*incRowB+j*incColB] *= alpha;
            }
        }
        alpha = 1.0;
    }

    for (j=0; j<mb; ++j) {
        mc     = (j!=mb-1 || _mc==0) ? MC : _mc;
        _alpha = (j==0) ? alpha : 1.0;

        pack_A(unitDiag, mc,
               &A[j*MC*incRowA+j*MC*incColA], incRowA, incColA,
               _A);

        for (k=0; k<nb; ++k) {
            nc = (k!=nb-1 || _nc==0) ? NC : _nc;

            pack_B(mc, nc,
                   &B[j*MC*incRowB+k*NC*incColB], incRowB, incColB,
                   _B);

            dtrsm_l_macro_kernel(mc, nc, _alpha);

            unpack_B(_B,
                     mc, nc,
                     &B[j*MC*incRowB+k*NC*incColB], incRowB, incColB);
        }

        if (m-MC*(j+1)>0) {
            dgemm_nn(m-MC*(j+1), n, mc, -1.0,
                     &A[(j+1)*MC*incRowA+j*MC*incColA], incRowA, incColA,
                     &B[j*MC*incRowB], incRowB, incColB,
                    alpha,
                    &B[(j+1)*MC*incRowB], incRowB, incColB);
        }
    }
    /*
    printf("X =\n");
    printMatrix(m, n, B, incRowB, incColB);
    */
}

