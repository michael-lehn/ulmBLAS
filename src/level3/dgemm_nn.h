#ifndef LEVEL3_DGEMM_NN_H
#define LEVEL3_DGEMM_NN_H 1

void
ULMBLAS(dgemm_nn)(const int         m,
                  const long         n,
                  const long         k,
                  const double      alpha,
                  const double      *A,
                  const long         ldA,
                  const double      *B,
                  const long         ldB,
                  const double      beta,
                  double            *C,
                  const long         ldC);

#endif // LEVEL3_DGEMM_NN_H
