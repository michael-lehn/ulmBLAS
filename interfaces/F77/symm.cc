#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/F77/config.h>
#include <interfaces/F77/xerbla.h>
#include <ulmblas/level3/sylmm.h>
#include <ulmblas/level3/syumm.h>

extern "C" {

void
F77BLAS(dsymm)(const char     *_side,
               const char     *_upLo,
               const int      *_m,
               const int      *_n,
               const double   *_alpha,
               const double   *A,
               const int      *_ldA,
               const double   *B,
               const int      *_ldB,
               const double   *_beta,
               double         *C,
               const int      *_ldC)
{
//
//  Dereference scalar parameters
//
    bool left     = (toupper(*_side) == 'L');
    bool lower    = (toupper(*_upLo) == 'L');
    int m         = *_m;
    int n         = *_n;
    double alpha  = *_alpha;
    int ldA       = *_ldA;
    double beta   = *_beta;
    int ldB       = *_ldB;
    int ldC       = *_ldC;

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
        F77BLAS(xerbla)("DSYMM ", &info);
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
