#ifndef LEVEL3_DGEMM_NN_H
#define LEVEL3_DGEMM_NN_H 1

#include <ulmblas.h>

//
//  Micro kernel for multiplying panels from A and B.
//  A points to a MR x kc panel and
//  B points to a kc x NR panel.
//
void
dgemm_micro_kernel(long kc,
                   double alpha, const double *A, const double *B,
                   double beta,
                   double *C, long incRowC, long incColC,
                   const double *nextA, const double *nextB);

//
//  Compute C <- beta*C + alpha*A*B
//
void
dgemm_nn(int            m,
         int            n,
         int            k,
         double         alpha,
         const double   *A,
         int            incRowA,
         int            incColA,
         const double   *B,
         int            incRowB,
         int            incColB,
         double         beta,
         double         *C,
         int            incRowC,
         int            incColC);

#endif // LEVEL3_DGEMM_NN_H
