#include <algorithm>
#include <cctype>
#include <complex>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sgemm)(const char     *transA_,
               const char     *transB_,
               const int      *m_,
               const int      *n_,
               const int      *k_,
               const float    *alpha_,
               const float    *A,
               const int      *ldA_,
               const float    *B,
               const int      *ldB_,
               const float    *beta_,
               float          *C,
               const int      *ldC_)
{
//
//  Dereference scalar parameters
//
    bool transA  = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool transB  = (toupper(*transB_) == 'T' || toupper(*transB_) == 'C');
    int m        = *m_;
    int n        = *n_;
    int k        = *k_;
    double alpha = *alpha_;
    int ldA      = *ldA_;
    int ldB      = *ldB_;
    double beta  = *beta_;
    int ldC      = *ldC_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (!transA) ? m : k;
    int numRowsB = (!transB) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

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
    } else if (ldA<std::max(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SGEMM ", &info);
    }

//
//  Start the operations.
//
    if (!transB) {
        if (!transA) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          false, A, 1, ldA,
                          false, B, 1, ldB,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm(m, n, k,
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
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          false, A, 1, ldA,
                          false, B, ldB, 1,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          false, A, ldA, 1,
                          false, B, ldB, 1,
                          beta,
                          C, 1, ldC);
        }
    }
}


void
F77BLAS(dgemm)(const char     *transA_,
               const char     *transB_,
               const int      *m_,
               const int      *n_,
               const int      *k_,
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
    bool transA  = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool transB  = (toupper(*transB_) == 'T' || toupper(*transB_) == 'C');
    int m        = *m_;
    int n        = *n_;
    int k        = *k_;
    double alpha = *alpha_;
    int ldA      = *ldA_;
    int ldB      = *ldB_;
    double beta  = *beta_;
    int ldC      = *ldC_;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (!transA) ? m : k;
    int numRowsB = (!transB) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

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
    } else if (ldA<std::max(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGEMM ", &info);
    }

//
//  Start the operations.
//
    if (!transB) {
        if (!transA) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          false, A, 1, ldA,
                          false, B, 1, ldB,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm(m, n, k,
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
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          false, A, 1, ldA,
                          false, B, ldB, 1,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          false, A, ldA, 1,
                          false, B, ldB, 1,
                          beta,
                          C, 1, ldC);
        }
    }
}

void
F77BLAS(cgemm)(const char     *transA_,
               const char     *transB_,
               const int      *m_,
               const int      *n_,
               const int      *k_,
               const float    *alpha_,
               const float    *A_,
               const int      *ldA_,
               const float    *B_,
               const int      *ldB_,
               const float    *beta_,
               float          *C_,
               const int      *ldC_)
{
    typedef std::complex<float> fcomplex;
//
//  Dereference scalar parameters
//
    bool transA  = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool transB  = (toupper(*transB_) == 'T' || toupper(*transB_) == 'C');
    bool conjA   = (toupper(*transA_) == 'R' || toupper(*transA_) == 'C');
    bool conjB   = (toupper(*transB_) == 'R' || toupper(*transB_) == 'C');
    int m        = *m_;
    int n        = *n_;
    int k        = *k_;
    int ldA      = *ldA_;
    int ldB      = *ldB_;
    int ldC      = *ldC_;

    fcomplex alpha(alpha_[0], alpha_[1]);
    fcomplex beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (!transA) ? m : k;
    int numRowsB = (!transB) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

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
    } else if (ldA<std::max(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CGEMM ", &info);
    }

//
//  Start the operations.
//
    if (!transB) {
        if (!transA) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          conjA, A, 1, ldA,
                          conjB, B, 1, ldB,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm(m, n, k,
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
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          conjA, A, 1, ldA,
                          conjB, B, ldB, 1,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          conjA, A, ldA, 1,
                          conjB, B, ldB, 1,
                          beta,
                          C, 1, ldC);
        }
    }
}


void
F77BLAS(zgemm)(const char     *transA_,
               const char     *transB_,
               const int      *m_,
               const int      *n_,
               const int      *k_,
               const double   *alpha_,
               const double   *A_,
               const int      *ldA_,
               const double   *B_,
               const int      *ldB_,
               const double   *beta_,
               double         *C_,
               const int      *ldC_)
{
    typedef std::complex<double> dcomplex;
//
//  Dereference scalar parameters
//
    bool transA  = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool transB  = (toupper(*transB_) == 'T' || toupper(*transB_) == 'C');
    bool conjA   = (toupper(*transA_) == 'R' || toupper(*transA_) == 'C');
    bool conjB   = (toupper(*transB_) == 'R' || toupper(*transB_) == 'C');
    int m        = *m_;
    int n        = *n_;
    int k        = *k_;
    int ldA      = *ldA_;
    int ldB      = *ldB_;
    int ldC      = *ldC_;

    dcomplex alpha(alpha_[0], alpha_[1]);
    dcomplex beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (!transA) ? m : k;
    int numRowsB = (!transB) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

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
    } else if (ldA<std::max(1,numRowsA)) {
        info = 8;
    } else if (ldB<std::max(1,numRowsB)) {
        info = 10;
    } else if (ldC<std::max(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZGEMM ", &info);
    }

//
//  Start the operations.
//
    if (!transB) {
        if (!transA) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          conjA, A, 1, ldA,
                          conjB, B, 1, ldB,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm(m, n, k,
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
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          conjA, A, 1, ldA,
                          conjB, B, ldB, 1,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          conjA, A, ldA, 1,
                          conjB, B, ldB, 1,
                          beta,
                          C, 1, ldC);
        }
    }
}


} // extern "C"
