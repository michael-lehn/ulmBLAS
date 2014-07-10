#include <ulmblas.h>
#include <math.h>

//
//  Solve A*X = alpha*B
//
//  where A is a upper triangular mxm matrix with unit or non-unit diagonal
//  and B is a general mxn matrix.
//

void
dtrsm_u(enum Diag       diag,
        int             m,
        int             n,
        double          alpha,
        const double    *A,
        int             incRowA,
        int             incColA,
        double          *B,
        int             incRowB,
        int             incColB)
{
    // Your implementation goes here
}
