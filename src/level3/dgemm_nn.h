#ifndef LEVEL3_DGEMM_NN_H
#define LEVEL3_DGEMM_NN_H 1

#include <ulmblas.h>

void
ULMBLAS(dgemm_nn)(int            m,
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
