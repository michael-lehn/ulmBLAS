#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/F77/config.h>
#include <interfaces/F77/xerbla.h>
#include <ulmblas/level3/trlmm.h>
#include <ulmblas/level3/trumm.h>

#include <ulmblas/auxiliary/printmatrix.h>

extern "C" {

void
F77BLAS(dtrmm)(const char     *_side,
               const char     *_upLo,
               const char     *_transA,
               const char     *_diag,
               const int      *_m,
               const int      *_n,
               const double   *_alpha,
               const double   *A,
               const int      *_ldA,
               double         *B,
               const int      *_ldB)
{
//
//  Dereference scalar parameters
//
    bool left     = (toupper(*_side) == 'L');
    bool lower    = (toupper(*_upLo) == 'L');
    bool transA   = (toupper(*_transA) == 'T' || toupper(*_transA) == 'C');
    bool unitDiag = (toupper(*_diag) == 'U');
    int m         = *_m;
    int n         = *_n;
    double alpha  = *_alpha;
    int ldA       = *_ldA;
    int ldB       = *_ldB;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*_side)!='L' && toupper(*_side)!='R') {
        info = 1;
    } else if (toupper(*_upLo)!='L' && toupper(*_upLo)!='U') {
        info = 2;
    } else if (toupper(*_transA)!='N' && toupper(*_transA)!='T'
            && toupper(*_transA)!='C' && toupper(*_transA)!='R')
    {
        info = 3;
    } else if (toupper(*_diag)!='U' && toupper(*_diag)!='N') {
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
        F77BLAS(xerbla)("DTRMM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            if (!transA) {
                ulmBLAS::trlmm(m, n, alpha, unitDiag, A, 1, ldA, B, 1, ldB);
            } else {
                ulmBLAS::trumm(m, n, alpha, unitDiag, A, ldA, 1, B, 1, ldB);
            }
        } else {
            if (!transA) {
                ulmBLAS::trumm(m, n, alpha, unitDiag, A, 1, ldA, B, 1, ldB);
            } else {
                ulmBLAS::trlmm(m, n, alpha, unitDiag, A, ldA, 1, B, 1, ldB);
            }
        }
    } else {
        if (lower) {
            if (!transA) {
                ulmBLAS::trumm(n, m, alpha, unitDiag, A, ldA, 1, B, ldB, 1);
            } else {
                ulmBLAS::trlmm(n, m, alpha, unitDiag, A, 1, ldA, B, ldB, 1);
            }
        } else {
            if (!transA) {
                ulmBLAS::trlmm(n, m, alpha, unitDiag, A, ldA, 1, B, ldB, 1);
            } else {
                ulmBLAS::trumm(n, m, alpha, unitDiag, A, 1, ldA, B, ldB, 1);
            }
        }
    }
}

} // extern "C"
