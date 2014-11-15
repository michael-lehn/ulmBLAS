#include <algorithm>
#include <cctype>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/level3/sylmm.h>
#include <ulmblas/level3/syumm.h>

extern "C" {

void
F77BLAS(dsymm)(const char     *side_,
               const char     *upLo_,
               const int      *m_,
               const int      *n_,
               const double   *alpha_,
               const double   *A,
               const int      *ldA_,
               const double   *B,
               const int      *ldB_,
               const double   *beta_,
               double         *C,
               const int      *ldC_)
{
//
//  Dereference scalar parameters
//
    bool left     = (toupper(*side_) == 'L');
    bool lower    = (toupper(*upLo_) == 'L');
    int m         = *m_;
    int n         = *n_;
    double alpha  = *alpha_;
    int ldA       = *ldA_;
    double beta   = *beta_;
    int ldB       = *ldB_;
    int ldC       = *ldC_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*side_)!='L' && toupper(*side_)!='R') {
        info = 1;
    } else if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
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
