#include <algorithm>
#include <interfaces/blas/C/config.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level1/copy.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level2/trlsv.h>
#include <ulmblas/level2/trusv.h>

extern "C" {

void
ULMBLAS(dtrsv)(enum UpLo     upLo,
               enum Trans    trans,
               enum Diag     diag,
               int           n,
               const double  *A,
               int           ldA,
               double        *x,
               int           incX)
{

//
//  Test the input parameters
//
    int info = 0;
    if (upLo!=Upper && upLo!=Lower) {
        info = 1;
    } else if (trans!=NoTrans && trans!=Trans
            && trans!=ConjTrans && trans!=Conj)
    {
        info = 2;
    } else if (diag!=NonUnit && diag!=Unit) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<std::max(1,n)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    }

    if (info!=0) {
        ULMBLAS(xerbla)("DTRSV ", &info);
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    bool unitDiag = (diag==Unit);

    if (upLo==Lower) {
        if (trans==NoTrans || trans==Conj) {
            ulmBLAS::trlsv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trusv(n, unitDiag, A, ldA, 1, x, incX);
        }
    } else {
        if (trans==NoTrans || trans==Conj) {
            ulmBLAS::trusv(n, unitDiag, A, 1, ldA, x, incX);
        } else {
            ulmBLAS::trlsv(n, unitDiag, A, ldA, 1, x, incX);
        }
    }
}

} // extern "C"
