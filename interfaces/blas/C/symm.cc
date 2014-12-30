#include BLAS_HEADER
#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level3/sylmm.h>
#include <ulmblas/level3/syumm.h>

#include <ulmblas/auxiliary/printmatrix.h>

extern "C" {

void
ULMBLAS(dsymm)(enum CBLAS_SIDE  side,
               enum CBLAS_UPLO  upLo,
               int              m,
               int              n,
               double           alpha,
               const double     *A,
               int              ldA,
               double           *B,
               int              ldB,
               double           beta,
               double           *C,
               int              ldC)
{
    bool left     = (side==CblasLeft);
    bool lower    = (upLo==CblasLower);

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

void
CBLAS(dsymm)(enum CBLAS_ORDER  order,
             enum CBLAS_SIDE   side,
             enum CBLAS_UPLO   upLo,
             int               m,
             int               n,
             double            alpha,
             const double      *A,
             int               ldA,
             double            *B,
             int               ldB,
             double            beta,
             double            *C,
             int               ldC)
{
    bool left = (side==CblasLeft);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int  numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    int info = 0;

    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (side!=CblasLeft && side!=CblasRight) {
        info = 2;
    } else if (upLo!=CblasLower && upLo!=CblasUpper) {
        info = 3;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 4;
            } else if (n<0) {
                info = 5;
            } else if (ldA<std::max(1,numRowsA)) {
                info = 8;
            } else if (ldB<std::max(1,m)) {
                info = 10;
            } else if (ldC<std::max(1,m)) {
                info = 13;
            }
        } else {
            if (n<0) {
                info = 4;
            } else if (m<0) {
                info = 5;
            } else if (ldA<std::max(1,numRowsA)) {
                info = 8;
            } else if (ldB<std::max(1,n)) {
                info = 10;
            } else if (ldC<std::max(1,n)) {
                info = 13;
            }
        }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_dsymm", "");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dsymm)(side, upLo, m, n, alpha, A, ldA, B, ldB, beta, C, ldC);
    } else {
        side = (side==CblasLeft) ? CblasRight : CblasLeft;
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dsymm)(side, upLo, n, m, alpha, A, ldA, B, ldB, beta, C, ldC);
    }
}

} // extern "C"
