/*
 *  verify-eigen: verify-large.c (campaign 008-2026-10-26-heim-avx-fma-memory)
 *  for Eigen.  For each n, C.noalias() += A*B with random A, B, C in
 *  [-1, 1] (column major, Eigen::Matrix<double, Dynamic, Dynamic,
 *  ColMajor>); ENTRIES randomly chosen elements of C are compared with the
 *  sum computed in long double, against the bound
 *  4*(k+2)*eps*(sum |A(i,l)*B(l,j)| + |C0(i,j)|).  Same random numbers
 *  as verify-large.
 *
 *      verify-eigen n1 n2 ...
 *
 *  Prints one line per n (max error / bound, number of violations).  Exit
 *  status 1 if any element violates the bound.
 */
#include <Eigen/Dense>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(EIGEN_USE_BLAS) || defined(EIGEN_USE_MKL_ALL) || defined(_OPENMP)
#error "this program checks Eigen's own single-threaded product"
#endif

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor>;

enum { ENTRIES = 2000 };

static double
uniform()
{
    return 2.0*rand()/RAND_MAX - 1.0;
}

int
main(int argc, char **argv)
{
    int bad = 0;
    srand(2026);
    for (int a=1; a<argc; ++a) {
        int    n = atoi(argv[a]);
        Matrix A(n, n), B(n, n), C(n, n), C0(n, n);
        // element order as in verify-large: A[i], B[i], C[i] for each i
        for (long i=0; i<(long)n*n; ++i) {
            A.data()[i] = uniform(); B.data()[i] = uniform();
            C.data()[i] = C0.data()[i] = uniform();
        }
        C.noalias() += A*B;
        double worst = 0;
        int    violations = 0;
        for (int e=0; e<ENTRIES; ++e) {
            int i = rand() % n, j = rand() % n;
            long double s = C0(i, j), bound = fabs(C0(i, j));
            for (int l=0; l<n; ++l) {
                long double p = (long double)A(i, l)*B(l, j);
                s += p;
                bound += fabsl(p);
            }
            bound *= 4.0L*(n+2)*DBL_EPSILON;
            long double err = fabsl(C(i, j) - s);
            if (!(err <= bound)) {
                ++violations;
            }
            if (err/bound > worst) {
                worst = (double)(err/bound);
            }
        }
        printf("n %4d: %d elements checked, max error/bound %.3g, %d violations\n",
               n, ENTRIES, worst, violations);
        bad |= violations>0;
    }
    return bad;
}
