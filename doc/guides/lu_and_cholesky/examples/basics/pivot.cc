// P A = L U with ulmBLAS::getrf, and what the pivot vector means: piv[i]
// is the row that was interchanged with row i, for i = 0, 1, ... in this
// order.  It is not the permutation itself.
#include "../common/util.h"
#include <ulmblas/lapack/getrf.h>

int
main()
{
    const long n = 4;
    // column major: the columns are (2 4 8 6), (1 3 7 7), ...
    double A[n * n] = {2, 4, 8, 6, 1, 3, 7, 7, 1, 3, 9, 9, 0, 1, 5, 8};
    double A0[n * n];
    for (long i = 0; i < n * n; ++i) {
	A0[i] = A[i];
    }
    print("A", n, n, A, n);

    long piv[n];
    const long info = ulmBLAS::getrf(n, n, A, 1L, n, piv, 1L);
    print("L and U in A", n, n, A, n);
    std::printf("info = %ld, piv = %ld %ld %ld %ld\n", info, piv[0], piv[1],
                piv[2], piv[3]);

    // the permutation: apply the interchanges to the numbers 0, ..., n-1
    long p[n] = {0, 1, 2, 3};
    for (long i = 0; i < n; ++i) {
	std::swap(p[i], p[piv[i]]);
    }
    std::printf("row i of P A is row p[i] of A: p = %ld %ld %ld %ld\n", p[0],
                p[1], p[2], p[3]);

    std::printf("||P A - L U|| / (n eps ||A||) = %.2f\n",
                luResidual(n, A0, A, piv));
}
