#include <interfaces/C/config.h>
#include <interfaces/C/xerbla.h>
#include <src/level3/gemm.h>

extern "C" {

void
ULMBLAS(dgemm)(const enum Trans  transA,
               const enum Trans  transB,
               const int         m,
               const int         n,
               const int         k,
               const double      alpha,
               const double      *A,
               const int         ldA,
               const double      *B,
               const int         ldB,
               const double      beta,
               double            *C,
               const int         ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (transA==NoTrans || transA==Conj) ? m : k;
    int numRowsB = (transB==NoTrans || transB==Conj) ? k : n;

//
//  Test the input parameters
//
    int info = 0;
    if (transA==0) {
        info = 1;
    } else if (transB==0) {
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
        ULMBLAS(xerbla)("DGEMM ", &info);
    }

//
//  Start the operations.
//
    if (transB==NoTrans || transB==Conj) {
        if (transA==NoTrans || transA==Conj) {
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
        if (transA==NoTrans || transA==Conj) {
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
