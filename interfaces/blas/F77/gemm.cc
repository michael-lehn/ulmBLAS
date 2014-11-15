#include <algorithm>
#include <cctype>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/level3/gemm.h>

extern "C" {

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
                          A, 1, ldA,
                          B, 1, ldB,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          A, ldA, 1,
                          B, 1, ldB,
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
                          A, 1, ldA,
                          B, ldB, 1,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          A, ldA, 1,
                          B, ldB, 1,
                          beta,
                          C, 1, ldC);
        }
    }
}

} // extern "C"
