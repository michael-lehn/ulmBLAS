#ifndef LEVEL3_DGEMM_NN_H
#define LEVEL3_DGEMM_NN_H 1

void
dgemm_nn(const int         m,
         const int         n,
         const int         k,
         const double      alpha,
         const double      *A,
         const int         ldA,
         const double      *B,
         const int         ldB,
         const double      beta,
         double            *C,
         const int         ldC);

#endif // LEVEL3_DGEMM_NN_H
