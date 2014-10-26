#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level3/sylrk.h>
#include <ulmblas/level3/syurk.h>

extern "C" {

void
ULMBLAS(dsyrk)(enum UpLo        upLo,
               enum Trans       trans,
               int              n,
               int              k,
               double           alpha,
               const double     *A,
               int              ldA,
               double           beta,
               double           *C,
               int              ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (trans==Trans || trans==ConjTrans) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

    if (upLo!=Lower && upLo!=Upper) {
        info = 1;
    } else if (trans!=NoTrans && trans!=Conj
            && trans!=Trans && trans!=ConjTrans)
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 6;
    } else if (ldC<std::max(1,n)) {
        info = 9;
    }

    if (info!=0) {
        ULMBLAS(xerbla)("DSYRK ", &info);
    }

//
//  Start the operations.
//
    if (trans==NoTrans || trans==Conj) {
        if (upLo==Lower) {
            ulmBLAS::sylrk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk(n, k, alpha, A, 1, ldA, beta, C, 1, ldC);
        }
    } else {
        if (upLo==Lower) {
            ulmBLAS::syurk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk(n, k, alpha, A, ldA, 1, beta, C, ldC, 1);
        }
    }
}

} // extern "C"
