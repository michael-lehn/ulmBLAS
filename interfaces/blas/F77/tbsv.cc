#include <algorithm>
#include <cctype>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(stbsv)(const char               *upLo_,
               const char               *transA_,
               const char               *diag_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
               const float              *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               float                    *x,
               const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    bool lowerA             = (toupper(*upLo_) == 'L');
    bool transA             = (toupper(*transA_) == 'T'
                            || toupper(*transA_) == 'C');
    bool unitDiag           = (toupper(*diag_) == 'U');
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE k    = *k_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 2;
    } else if (toupper(*diag_)!='U' && toupper(*diag_)!='N') {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,k+1)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("STBSV ", &info, 6);
        return;
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tblsv<std::ptrdiff_t>(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbusv<std::ptrdiff_t>(n, k, unitDiag, A, ldA, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tblstv<std::ptrdiff_t>(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbustv<std::ptrdiff_t>(n, k, unitDiag, A, ldA, x, incX);
        }
    }
}

void
F77BLAS(dtbsv)(const char               *upLo_,
               const char               *transA_,
               const char               *diag_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
               const double             *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               double                   *x,
               const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    bool lowerA             = (toupper(*upLo_) == 'L');
    bool transA             = (toupper(*transA_) == 'T'
                            || toupper(*transA_) == 'C');
    bool unitDiag           = (toupper(*diag_) == 'U');
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE k    = *k_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 2;
    } else if (toupper(*diag_)!='U' && toupper(*diag_)!='N') {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,k+1)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DTBSV ", &info, 6);
        return;
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tblsv<std::ptrdiff_t>(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbusv<std::ptrdiff_t>(n, k, unitDiag, A, ldA, x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tblstv<std::ptrdiff_t>(n, k, unitDiag, A, ldA, x, incX);
        } else {
            ulmBLAS::tbustv<std::ptrdiff_t>(n, k, unitDiag, A, ldA, x, incX);
        }
    }
}

void
F77BLAS(ctbsv)(const char               *upLo_,
               const char               *transA_,
               const char               *diag_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
               const float              *A_,
               const ULMBLAS_INDEX_TYPE *ldA_,
               float                    *x_,
               const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    bool lowerA             = (toupper(*upLo_) == 'L');
    bool transA             = (toupper(*transA_) == 'T'
                            || toupper(*transA_) == 'C');
    bool unitDiag           = (toupper(*diag_) == 'U');
    bool conjA              = toupper(*transA_) == 'C';
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE k    = *k_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<float> fcomplex;
    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *x = reinterpret_cast<fcomplex *>(x_);

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 2;
    } else if (toupper(*diag_)!='U' && toupper(*diag_)!='N') {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,k+1)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CTBSV ", &info, 6);
        return;
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tblsv<std::ptrdiff_t>(n, k, unitDiag, conjA, A, ldA,
                                           x, incX);
        } else {
            ulmBLAS::tbusv<std::ptrdiff_t>(n, k, unitDiag, conjA, A, ldA,
                                           x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tblstv<std::ptrdiff_t>(n, k, unitDiag, conjA, A, ldA,
                                            x, incX);
        } else {
            ulmBLAS::tbustv<std::ptrdiff_t>(n, k, unitDiag, conjA, A, ldA,
                                            x, incX);
        }
    }
}

void
F77BLAS(ztbsv)(const char               *upLo_,
               const char               *transA_,
               const char               *diag_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
               const double             *A_,
               const ULMBLAS_INDEX_TYPE *ldA_,
               double                   *x_,
               const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    bool lowerA             = (toupper(*upLo_) == 'L');
    bool transA             = (toupper(*transA_) == 'T'
                            || toupper(*transA_) == 'C');
    bool unitDiag           = (toupper(*diag_) == 'U');
    bool conjA              = toupper(*transA_) == 'C';
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE k    = *k_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<double> dcomplex;
    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *x = reinterpret_cast<dcomplex *>(x_);

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (toupper(*transA_)!='N' && toupper(*transA_)!='T'
     && toupper(*transA_)!='C' && toupper(*transA_)!='R')
    {
        info = 2;
    } else if (toupper(*diag_)!='U' && toupper(*diag_)!='N') {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,k+1)) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZTBSV ", &info, 6);
        return;
    }

    if (incX<0) {
        x -= incX*(n-1);
    }

//
//  Start the operations.
//
    if (!transA) {
        if (lowerA) {
            ulmBLAS::tblsv<std::ptrdiff_t>(n, k, unitDiag, conjA, A, ldA,
                                           x, incX);
        } else {
            ulmBLAS::tbusv<std::ptrdiff_t>(n, k, unitDiag, conjA, A, ldA,
                                           x, incX);
        }
    } else {
        if (lowerA) {
            ulmBLAS::tblstv<std::ptrdiff_t>(n, k, unitDiag, conjA, A, ldA,
                                            x, incX);
        } else {
            ulmBLAS::tbustv<std::ptrdiff_t>(n, k, unitDiag, conjA, A, ldA,
                                            x, incX);
        }
    }
}

} // extern "C"
