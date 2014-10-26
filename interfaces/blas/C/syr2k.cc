#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level3/sylr2k.h>
#include <ulmblas/level3/syur2k.h>

extern "C" {

void
ULMBLAS(dsyr2k)(enum UpLo        upLo,
                enum Trans       trans,
                int              n,
                int              k,
                double           alpha,
                const double     *A,
                int              ldA,
                const double     *B,
                int              ldB,
                double           beta,
                double           *C,
                int              ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (trans==Trans || trans==ConjTrans) ? k : n;
    int numRowsB = (trans==Trans || trans==ConjTrans) ? k : n;

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
    } else if (ldB<std::max(1,numRowsB)) {
        info = 8;
    } else if (ldC<std::max(1,n)) {
        info = 11;
    }

    if (info!=0) {
        ULMBLAS(xerbla)("DSYR2K", &info);
    }

//
//  Start the operations.
//
    if (trans==NoTrans || trans==Conj) {
        if (upLo==Lower) {
            ulmBLAS::sylr2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        } else {
            ulmBLAS::syur2k(n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
        }
    } else {
        if (upLo==Lower) {
            ulmBLAS::syur2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        } else {
            ulmBLAS::sylr2k(n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
        }
    }
}

} // extern "C"
