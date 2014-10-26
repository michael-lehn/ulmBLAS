#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/config.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level3/sylmm.h>
#include <ulmblas/level3/syumm.h>

#include <ulmblas/auxiliary/printmatrix.h>

extern "C" {

void
ULMBLAS(dsymm)(const enum Side  side,
               const enum UpLo  upLo,
               const int        m,
               const int        n,
               const double     alpha,
               const double     *A,
               const int        ldA,
               double           *B,
               const int        ldB,
               const double     beta,
               double           *C,
               const int        ldC)
{
//
//  Dereference scalar parameters
//
    bool left     = (side==Left);
    bool lower    = (upLo==Lower);

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
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max(1,m)) {
        info = 9;
    } else if (ldC<std::max(1,m)) {
        info = 12;
    }

    if (info!=0) {
        ULMBLAS(xerbla)("DSYMM ", &info);
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            ulmBLAS::sylmm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::syumm(m, n, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syumm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylmm(n, m, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}

} // extern "C"
