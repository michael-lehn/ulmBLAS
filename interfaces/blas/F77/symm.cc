#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(ssymm)(const char               *side_,
               const char               *upLo_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const float              *B,
               const ULMBLAS_INDEX_TYPE *ldB_,
               const float              *beta_,
               float                    *C,
               const ULMBLAS_INDEX_TYPE *ldC_)
{
//
//  Dereference scalar parameters
//
    bool left              = (toupper(*side_) == 'L');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    double alpha           = *alpha_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    double beta            = *beta_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*side_)!='L' && toupper(*side_)!='R') {
        info = 1;
    } else if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SSYMM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            ulmBLAS::sylmm<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, B, 1, ldB,
                                           beta, C, 1, ldC);
        } else {
            ulmBLAS::syumm<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, B, 1, ldB,
                                           beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syumm<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, B, ldB, 1,
                                           beta, C, ldC, 1);
        } else {
            ulmBLAS::sylmm<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, B, ldB, 1,
                                           beta, C, ldC, 1);
        }
    }
}

void
F77BLAS(dsymm)(const char               *side_,
               const char               *upLo_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *A,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const double             *B,
               const ULMBLAS_INDEX_TYPE *ldB_,
               const double             *beta_,
               double                   *C,
               const ULMBLAS_INDEX_TYPE *ldC_)
{
//
//  Dereference scalar parameters
//
    bool left              = (toupper(*side_) == 'L');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    double alpha           = *alpha_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    double beta            = *beta_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*side_)!='L' && toupper(*side_)!='R') {
        info = 1;
    } else if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DSYMM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            ulmBLAS::sylmm<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, B, 1, ldB,
                                           beta, C, 1, ldC);
        } else {
            ulmBLAS::syumm<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, B, 1, ldB,
                                           beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syumm<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, B, ldB, 1,
                                           beta, C, ldC, 1);
        } else {
            ulmBLAS::sylmm<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, B, ldB, 1,
                                           beta, C, ldC, 1);
        }
    }
}

void
F77BLAS(csymm)(const char               *side_,
               const char               *upLo_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *A_,
               const ULMBLAS_INDEX_TYPE *ldA_,
               const float              *B_,
               const ULMBLAS_INDEX_TYPE *ldB_,
               const float              *beta_,
               float                    *C_,
               const ULMBLAS_INDEX_TYPE *ldC_)
{
    typedef std::complex<float> fcomplex;
//
//  Dereference scalar parameters
//
    bool left              = (toupper(*side_) == 'L');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

    fcomplex alpha(alpha_[0], alpha_[1]);
    fcomplex beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*side_)!='L' && toupper(*side_)!='R') {
        info = 1;
    } else if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CSYMM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            ulmBLAS::sylmm<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, B, 1, ldB,
                                           beta, C, 1, ldC);
        } else {
            ulmBLAS::syumm<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, B, 1, ldB,
                                           beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syumm<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, B, ldB, 1,
                                           beta, C, ldC, 1);
        } else {
            ulmBLAS::sylmm<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, B, ldB, 1,
                                           beta, C, ldC, 1);
        }
    }
}


void
F77BLAS(zsymm)(const char               *side_,
               const char               *upLo_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
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
    bool left              = (toupper(*side_) == 'L');
    bool lower             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

    dcomplex alpha(alpha_[0], alpha_[1]);
    dcomplex beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (left) ? m : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*side_)!='L' && toupper(*side_)!='R') {
        info = 1;
    } else if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 9;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZSYMM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (left) {
        if (lower) {
            ulmBLAS::sylmm<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, B, 1, ldB,
                                           beta, C, 1, ldC);
        } else {
            ulmBLAS::syumm<std::ptrdiff_t>(m, n, alpha, A, 1, ldA, B, 1, ldB,
                                           beta, C, 1, ldC);
        }
    } else {
        if (lower) {
            ulmBLAS::syumm<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, B, ldB, 1,
                                           beta, C, ldC, 1);
        } else {
            ulmBLAS::sylmm<std::ptrdiff_t>(n, m, alpha, A, ldA, 1, B, ldB, 1,
                                           beta, C, ldC, 1);
        }
    }
}

} // extern "C"
