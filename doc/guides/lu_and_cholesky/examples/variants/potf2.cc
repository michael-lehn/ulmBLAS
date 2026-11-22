// The unblocked Cholesky factorization of ulmBLAS (potf2) chooses its
// variant by the storage of the matrix: left-looking (crout) if the
// columns are contiguous, row by row (rows) if the rows are, right-looking
// otherwise.  Here each of them runs on the storage it is made for, and
// right-looking on both, for comparison.
#include "../common/util.h"
#include <ulmblas/lapack/potf2.h>
#include <vector>

int
main()
{
    const long n = 1500;
    std::vector<double> A0(n * n), W(n * n), A(n * n);
    spdMatrix(n, A0.data(), W.data()); // symmetric: also row major

    const double flops = double(n) * n * n / 3;
    namespace impl = ulmBLAS::potf2_impl;

    std::printf("n = %ld\n", n);
    A = A0;
    lap();
    impl::crout(n, A.data(), n);
    std::printf("column major, left-looking (crout):  %5.2f GFLOPS\n",
                1e-9 * flops / lap());
    A = A0;
    lap();
    impl::rightLooking(n, A.data(), 1L, n);
    std::printf("column major, right-looking:         %5.2f GFLOPS\n",
                1e-9 * flops / lap());
    A = A0;
    lap();
    impl::rows(n, A.data(), n);
    std::printf("row major,    row by row (rows):     %5.2f GFLOPS\n",
                1e-9 * flops / lap());
    A = A0;
    lap();
    impl::rightLooking(n, A.data(), n, 1L);
    std::printf("row major,    right-looking:         %5.2f GFLOPS\n",
                1e-9 * flops / lap());
}
