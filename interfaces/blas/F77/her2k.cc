#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(cher2k)(const char               *upLo_,
                const char               *trans_,
                const ULMBLAS_INDEX_TYPE *n_,
                const ULMBLAS_INDEX_TYPE *k_,
                const float              *alpha_,
                const float              *A_,
                const ULMBLAS_INDEX_TYPE *ldA_,
                const float              *B_,
                const ULMBLAS_INDEX_TYPE *ldB_,
                const float              *beta_,
                float                    *C_,
                const ULMBLAS_INDEX_TYPE *ldC_)
{
    typedef std::complex<float > fcomplex;
//
//  Dereference scalar parameters
//
    bool trans             = (toupper(*trans_) == 'C');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    float beta             = *beta_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

    fcomplex  alpha(alpha_[0], alpha_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (trans) ? k : n;
    ULMBLAS_INDEX_TYPE numRowsB = (trans) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (toupper(*trans_)!='N' && toupper(*trans_)!='C')
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
        info = 9;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CHER2K", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::helr2k<std::ptrdiff_t>(n, k, alpha,
                                            A, 1, ldA,
                                            B, 1, ldB,
                                            beta,
                                            C, 1, ldC);
        } else {
            ulmBLAS::heur2k<std::ptrdiff_t>(n, k, alpha,
                                            A, 1, ldA,
                                            B, 1, ldB,
                                            beta,
                                            C, 1, ldC);
        }
    } else {
        alpha = ulmBLAS::conjugate(alpha);
        if (lower) {
            ulmBLAS::heur2k<std::ptrdiff_t>(n, k, alpha,
                                            A, ldA, 1,
                                            B, ldB, 1,
                                            beta,
                                            C, ldC, 1);
        } else {
            ulmBLAS::helr2k<std::ptrdiff_t>(n, k, alpha,
                                            A, ldA, 1,
                                            B, ldB, 1,
                                            beta, C,
                                            ldC, 1);
        }
    }
}


void
F77BLAS(zher2k)(const char               *upLo_,
                const char               *trans_,
                const ULMBLAS_INDEX_TYPE *n_,
                const ULMBLAS_INDEX_TYPE *k_,
                const double             *alpha_,
                const double             *A_,
                const ULMBLAS_INDEX_TYPE *ldA_,
                const double             *B_,
                const ULMBLAS_INDEX_TYPE *ldB_,
                const double             *beta_,
                double                   *C_,
                const ULMBLAS_INDEX_TYPE *ldC_)
{
    typedef std::complex<double> dcomplex;
//
//  Dereference scalar parameters
//
    bool trans             = (toupper(*trans_) == 'C');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    double beta            = *beta_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

    dcomplex  alpha(alpha_[0], alpha_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (trans) ? k : n;
    ULMBLAS_INDEX_TYPE numRowsB = (trans) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (toupper(*trans_)!='N' && toupper(*trans_)!='C')
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
        info = 9;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZHER2K", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::helr2k<std::ptrdiff_t>(n, k, alpha,
                                            A, 1, ldA,
                                            B, 1, ldB,
                                            beta,
                                            C, 1, ldC);
        } else {
            ulmBLAS::heur2k<std::ptrdiff_t>(n, k, alpha,
                                            A, 1, ldA,
                                            B, 1, ldB,
                                            beta,
                                            C, 1, ldC);
        }
    } else {
        alpha = ulmBLAS::conjugate(alpha);
        if (lower) {
            ulmBLAS::heur2k<std::ptrdiff_t>(n, k, alpha,
                                            A, ldA, 1,
                                            B, ldB, 1,
                                            beta,
                                            C, ldC, 1);
        } else {
            ulmBLAS::helr2k<std::ptrdiff_t>(n, k, alpha,
                                            A, ldA, 1,
                                            B, ldB, 1,
                                            beta, C,
                                            ldC, 1);
        }
    }
}


} // extern "C"
