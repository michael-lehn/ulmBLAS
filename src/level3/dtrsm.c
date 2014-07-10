#include <ulmblas.h>
#include <auxiliary/xerbla.h>
#include <math.h>
#include <stdio.h>
#include <level3/dtrsm_l.h>
#include <level3/dtrsm_u.h>

void
ULMBLAS(dtrsm)(enum Side      side,
               enum UpLo      upLo,
               enum Trans     transA,
               enum Diag      diag,
               int            m,
               int            n,
               double         alpha,
               const double   *A,
               int            ldA,
               double         *B,
               int            ldB)
{
    if (side==Left) {
        if (transA==NoTrans) {
            if (upLo==Upper) {
                dtrsm_u(diag, m, n, alpha, A, 1, ldA, B, 1, ldB);
            } else {
                dtrsm_l(diag, m, n, alpha, A, 1, ldA, B, 1, ldB);
            }
        } else {
            if (upLo==Upper) {
                dtrsm_l(diag, m, n, alpha, A, ldA, 1, B, 1, ldB);
            } else {
                dtrsm_u(diag, m, n, alpha, A, ldA, 1, B, 1, ldB);
            }
        }
    } else {
        if (transA==NoTrans) {
            if (upLo==Upper) {
                dtrsm_l(diag, n, m, alpha, A, ldA, 1, B, ldB, 1);
            } else {
                dtrsm_u(diag, n, m, alpha, A, ldA, 1, B, ldB, 1);
            }
        } else {
            if (upLo==Upper) {
                dtrsm_u(diag, n, m, alpha, A, 1, ldA, B, ldB, 1);
            } else {
                dtrsm_l(diag, n, m, alpha, A, 1, ldA, B, ldB, 1);
            }
        }
    }
}

void
F77BLAS(dtrsm)(const char    *_side,
               const char    *_upLo,
               const char    *_transA,
               const char    *_diag,
               int           *_m,
               int           *_n,
               double        *_alpha,
               const double  *A,
               int           *_ldA,
               double        *B,
               int           *_ldB)
{
//
//  Dereference scalar parameters
//
    enum Side   side   = charToSide(*_side);
    enum UpLo   upLo   = charToUpLo(*_upLo);
    enum Trans  transA = charToTranspose(*_transA);
    enum Diag   diag   = charToDiag(*_diag);
    int m             = *_m;
    int n             = *_n;
    double alpha      = *_alpha;
    int ldA           = *_ldA;
    int ldB           = *_ldB;

//
//  Set  numRowsA as the number of rows of A
//
    int numRowsA = (side==Left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;
    if (side==0) {
        info = 1;
    } else if (upLo==0) {
        info = 2;
    } else if (transA==0) {
        info = 3;
    } else if (diag==0) {
        info = 4;
    } else if (m<0) {
        info = 5;
    } else if (n<0) {
        info = 6;
    } else if (ldA<max(1,numRowsA)) {
        info = 9;
    } else if (ldB<max(1,m)) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DTRSM ", &info);
    }

    ULMBLAS(dtrsm)(side, upLo, transA, diag,
                   m, n,
                   alpha,
                   A, ldA,
                   B, ldB);
}
