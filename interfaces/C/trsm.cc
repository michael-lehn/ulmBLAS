#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/C/config.h>
#include <interfaces/C/xerbla.h>
#include <ulmblas/level3/trlsm.h>
#include <ulmblas/level3/trusm.h>

#include <ulmblas/auxiliary/printmatrix.h>

extern "C" {

void
ULMBLAS(dtrsm)(const enum Side  side,
               const enum UpLo  upLo,
               const enum Trans transA,
               const enum Diag  diag,
               const int        m,
               const int        n,
               const double     alpha,
               const double     *A,
               const int        ldA,
               double           *B,
               const int        ldB)
{
//
//  Dereference scalar parameters
//
    bool left     = (side==Left);
    bool lower    = (upLo==Lower);
    bool trans    = (transA==Trans || transA==ConjTrans);
    bool unitDiag = (diag==Unit);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (side!=Left && side!=Right) {
        info = 1;
    } else if (upLo!=Lower && upLo!=Upper) {
        info = 2;
    } else if (transA!=NoTrans && transA!=Trans
            && transA!=Conj && transA!=ConjTrans)
    {
        info = 3;
    } else if (diag!=Unit && diag!=NonUnit) {
        info = 4;
    } else if (m<0) {
        info = 5;
    } else if (n<0) {
        info = 6;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 9;
    } else if (ldB<std::max(1,m)) {
        info = 11;
    }

    if (info!=0) {
        ULMBLAS(xerbla)("DTRMM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!trans) {
                ulmBLAS::trlsm(m, n, alpha, unitDiag, A, 1, ldA, B, 1, ldB);
            } else {
                ulmBLAS::trusm(m, n, alpha, unitDiag, A, ldA, 1, B, 1, ldB);
            }
        } else {
            if (!trans) {
                ulmBLAS::trusm(m, n, alpha, unitDiag, A, 1, ldA, B, 1, ldB);
            } else {
                ulmBLAS::trlsm(m, n, alpha, unitDiag, A, ldA, 1, B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!trans) {
                ulmBLAS::trusm(n, m, alpha, unitDiag, A, ldA, 1, B, ldB, 1);
            } else {
                ulmBLAS::trlsm(n, m, alpha, unitDiag, A, 1, ldA, B, ldB, 1);
            }
        } else {
            if (!trans) {
                ulmBLAS::trlsm(n, m, alpha, unitDiag, A, ldA, 1, B, ldB, 1);
            } else {
                ulmBLAS::trusm(n, m, alpha, unitDiag, A, 1, ldA, B, ldB, 1);
            }
        }
    }
}

} // extern "C"
