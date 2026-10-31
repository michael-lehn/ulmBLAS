#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(ssyrk)(const char               *upLo_,
               const char               *trans_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
               const float              *alpha_,
               const float              *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const float              *beta_,
               float                    *C,
               const ULMBLAS_INDEX_TYPE *ldC_)
{
//
//  Dereference scalar parameters
//
    bool trans             = (toupper(*trans_) == 'T'
                           || toupper(*trans_) == 'C');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
    float  alpha           = *alpha_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    float  beta            = *beta_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (trans) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (toupper(*trans_)!='N'
            && toupper(*trans_)!='T'
            && toupper(*trans_)!='C'
            && toupper(*trans_)!='R')
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 10;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SSYRK ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::sylrk<std::ptrdiff_t>(n, k, alpha, A, 1, ldA,
                                           beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk<std::ptrdiff_t>(n, k, alpha, A, 1, ldA,
                                           beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syurk<std::ptrdiff_t>(n, k, alpha, A, ldA, 1,
                                           beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk<std::ptrdiff_t>(n, k, alpha, A, ldA, 1,
                                           beta, C, ldC, 1);
        }
    }
}


void
F77BLAS(dsyrk)(const char               *upLo_,
               const char               *trans_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
               const double             *alpha_,
               const double             *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const double             *beta_,
               double                   *C,
               const ULMBLAS_INDEX_TYPE *ldC_)
{
//
//  Dereference scalar parameters
//
    bool trans             = (toupper(*trans_) == 'T'
                           || toupper(*trans_) == 'C');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
    double alpha           = *alpha_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    double beta            = *beta_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (trans) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (toupper(*trans_)!='N'
            && toupper(*trans_)!='T'
            && toupper(*trans_)!='C'
            && toupper(*trans_)!='R')
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 10;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DSYRK ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::sylrk<std::ptrdiff_t>(n, k, alpha, A, 1, ldA,
                                           beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk<std::ptrdiff_t>(n, k, alpha, A, 1, ldA,
                                           beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syurk<std::ptrdiff_t>(n, k, alpha, A, ldA, 1,
                                           beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk<std::ptrdiff_t>(n, k, alpha, A, ldA, 1,
                                           beta, C, ldC, 1);
        }
    }
}

void
F77BLAS(csyrk)(const char               *upLo_,
               const char               *trans_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
               const float              *alpha_,
               const float              *A_,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const float              *beta_,
               float                    *C_,
               const ULMBLAS_INDEX_TYPE *ldC_)
{
    typedef std::complex<float> fcomplex;
//
//  Dereference scalar parameters
//
    bool trans             = (toupper(*trans_) == 'T');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

    fcomplex alpha(alpha_[0], alpha_[1]);
    fcomplex beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (trans) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (toupper(*trans_)!='N' && toupper(*trans_)!='T')
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 10;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CSYRK ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::sylrk<std::ptrdiff_t>(n, k, alpha, A, 1, ldA,
                                           beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk<std::ptrdiff_t>(n, k, alpha, A, 1, ldA,
                                           beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syurk<std::ptrdiff_t>(n, k, alpha, A, ldA, 1,
                                           beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk<std::ptrdiff_t>(n, k, alpha, A, ldA, 1,
                                           beta, C, ldC, 1);
        }
    }
}


void
F77BLAS(zsyrk)(const char               *upLo_,
               const char               *trans_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
               const double             *alpha_,
               const double             *A_,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const double             *beta_,
               double                   *C_,
               const ULMBLAS_INDEX_TYPE *ldC_)
{
    typedef std::complex<double> dcomplex;
//
//  Dereference scalar parameters
//
    bool trans             = (toupper(*trans_) == 'T');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

    dcomplex alpha(alpha_[0], alpha_[1]);
    dcomplex beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (trans) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (toupper(*trans_)!='N' && toupper(*trans_)!='T')
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 10;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZSYRK ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::sylrk<std::ptrdiff_t>(n, k, alpha, A, 1, ldA,
                                           beta, C, 1, ldC);
        } else {
            ulmBLAS::syurk<std::ptrdiff_t>(n, k, alpha, A, 1, ldA,
                                           beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syurk<std::ptrdiff_t>(n, k, alpha, A, ldA, 1,
                                           beta, C, ldC, 1);
        } else {
            ulmBLAS::sylrk<std::ptrdiff_t>(n, k, alpha, A, ldA, 1,
                                           beta, C, ldC, 1);
        }
    }
}

} // extern "C"
