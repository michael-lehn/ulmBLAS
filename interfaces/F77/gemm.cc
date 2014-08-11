#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/F77/config.h>
#include <interfaces/F77/xerbla.h>
#include <src/level3/gemm.h>
#include <src/level3/gemm.tcc>

extern "C" {

void
F77BLAS(dgemm)(const char     *_transA,
               const char     *_transB,
               const int      *_m,
               const int      *_n,
               const int      *_k,
               const double   *_alpha,
               const double   *A,
               const int      *_ldA,
               const double   *B,
               const int      *_ldB,
               const double   *_beta,
               double         *C,
               const int      *_ldC)
{
//
//  Dereference scalar parameters
//
    bool transA  = (toupper(*_transA) == 'T' || toupper(*_transA) == 'C');
    bool transB  = (toupper(*_transB) == 'T' || toupper(*_transB) == 'C');
    int m        = *_m;
    int n        = *_n;
    int k        = *_k;
    double alpha = *_alpha;
    int ldA      = *_ldA;
    int ldB      = *_ldB;
    double beta  = *_beta;
    int ldC      = *_ldC;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (!transA) ? m : k;
    int numRowsB = (!transB) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*_transA)!='N'
     && toupper(*_transA)!='T'
     && toupper(*_transA)!='C'
     && toupper(*_transA)!='R')
    {
        info = 1;
    } else if (toupper(*_transB)!='N'
            && toupper(*_transB)!='T'
            && toupper(*_transB)!='C'
            && toupper(*_transB)!='R')
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
