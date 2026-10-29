#include <algorithm>
#include <cctype>
#include <complex>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sgemm)(const char               *transA_,
               const char               *transB_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
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
    bool transA            = (toupper(*transA_) == 'T'
                           || toupper(*transA_) == 'C');
    bool transB            = (toupper(*transB_) == 'T'
                           || toupper(*transB_) == 'C');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
    double alpha           = *alpha_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    double beta            = *beta_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (!transA) ? m : k;
    ULMBLAS_INDEX_TYPE numRowsB = (!transB) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*transA_)!='N'
     && toupper(*transA_)!='T'
     && toupper(*transA_)!='C'
     && toupper(*transA_)!='R')
    {
        info = 1;
    } else if (toupper(*transB_)!='N'
            && toupper(*transB_)!='T'
            && toupper(*transB_)!='C'
            && toupper(*transB_)!='R')
    {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SGEMM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!transB) {
        if (!transA) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, 1, ldA,
                                          false, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, ldA, 1,
                                          false, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        }
    } else {
        if (!transA) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, 1, ldA,
                                          false, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, ldA, 1,
                                          false, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        }
    }
}


void
F77BLAS(dgemm)(const char               *transA_,
               const char               *transB_,
               const ULMBLAS_INDEX_TYPE *m_,
               const ULMBLAS_INDEX_TYPE *n_,
               const ULMBLAS_INDEX_TYPE *k_,
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
    bool transA            = (toupper(*transA_) == 'T'
                           || toupper(*transA_) == 'C');
    bool transB            = (toupper(*transB_) == 'T'
                           || toupper(*transB_) == 'C');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
    double alpha           = *alpha_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    double beta            = *beta_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    ULMBLAS_INDEX_TYPE numRowsA = (!transA) ? m : k;
    ULMBLAS_INDEX_TYPE numRowsB = (!transB) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*transA_)!='N'
     && toupper(*transA_)!='T'
     && toupper(*transA_)!='C'
     && toupper(*transA_)!='R')
    {
        info = 1;
    } else if (toupper(*transB_)!='N'
            && toupper(*transB_)!='T'
            && toupper(*transB_)!='C'
            && toupper(*transB_)!='R')
    {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGEMM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!transB) {
        if (!transA) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, 1, ldA,
                                          false, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, ldA, 1,
                                          false, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        }
    } else {
        if (!transA) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, 1, ldA,
                                          false, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          false, A, ldA, 1,
                                          false, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        }
    }
}

void
F77BLAS(cgemm)(const char               *transA_,
               const char               *transB_,
               const ULMBLAS_INDEX_TYPE *m_,
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
    typedef std::complex<float> fcomplex;
//
//  Dereference scalar parameters
//
    bool transA            = (toupper(*transA_) == 'T'
                           || toupper(*transA_) == 'C');
    bool transB            = (toupper(*transB_) == 'T'
                           || toupper(*transB_) == 'C');
    bool conjA             = (toupper(*transA_) == 'R'
                           || toupper(*transA_) == 'C');
    bool conjB             = (toupper(*transB_) == 'R'
                           || toupper(*transB_) == 'C');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
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
    ULMBLAS_INDEX_TYPE numRowsA = (!transA) ? m : k;
    ULMBLAS_INDEX_TYPE numRowsB = (!transB) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*transA_)!='N'
     && toupper(*transA_)!='T'
     && toupper(*transA_)!='C'
     && toupper(*transA_)!='R')
    {
        info = 1;
    } else if (toupper(*transB_)!='N'
            && toupper(*transB_)!='T'
            && toupper(*transB_)!='C'
            && toupper(*transB_)!='R')
    {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CGEMM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!transB) {
        if (!transA) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, 1, ldA,
                                          conjB, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, ldA, 1,
                                          conjB, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        }
    } else {
        if (!transA) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, 1, ldA,
                                          conjB, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, ldA, 1,
                                          conjB, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        }
    }
}


void
F77BLAS(zgemm)(const char               *transA_,
               const char               *transB_,
               const ULMBLAS_INDEX_TYPE *m_,
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
    bool transA            = (toupper(*transA_) == 'T'
                           || toupper(*transA_) == 'C');
    bool transB            = (toupper(*transB_) == 'T'
                           || toupper(*transB_) == 'C');
    bool conjA             = (toupper(*transA_) == 'R'
                           || toupper(*transA_) == 'C');
    bool conjB             = (toupper(*transB_) == 'R'
                           || toupper(*transB_) == 'C');
    ULMBLAS_INDEX_TYPE m   = *m_;
    ULMBLAS_INDEX_TYPE n   = *n_;
    ULMBLAS_INDEX_TYPE k   = *k_;
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
    ULMBLAS_INDEX_TYPE numRowsA = (!transA) ? m : k;
    ULMBLAS_INDEX_TYPE numRowsB = (!transB) ? k : n;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*transA_)!='N'
     && toupper(*transA_)!='T'
     && toupper(*transA_)!='C'
     && toupper(*transA_)!='R')
    {
        info = 1;
    } else if (toupper(*transB_)!='N'
            && toupper(*transB_)!='T'
            && toupper(*transB_)!='C'
            && toupper(*transB_)!='R')
    {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max<ULMBLAS_INDEX_TYPE>(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max<ULMBLAS_INDEX_TYPE>(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZGEMM ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (!transB) {
        if (!transA) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, 1, ldA,
                                          conjB, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, ldA, 1,
                                          conjB, B, 1, ldB,
                                          beta,
                                          C, 1, ldC);
        }
    } else {
        if (!transA) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, 1, ldA,
                                          conjB, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                          alpha,
                                          conjA, A, ldA, 1,
                                          conjB, B, ldB, 1,
                                          beta,
                                          C, 1, ldC);
        }
    }
}


} // extern "C"
