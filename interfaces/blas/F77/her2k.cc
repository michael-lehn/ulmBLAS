#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level3/helr2k.h>
#include <ulmblas/level3/heur2k.h>

extern "C" {

void
F77BLAS(zher2k)(const char     *upLo_,
                const char     *trans_,
                const int      *n_,
                const int      *k_,
                const double   *alpha_,
                const double   *A_,
                const int      *ldA_,
                const double   *B_,
                const int      *ldB_,
                const double   *beta_,
                double         *C_,
                const int      *ldC_)
{
    typedef std::complex<double> dcomplex;
//
//  Dereference scalar parameters
//
    bool trans   = (toupper(*trans_) == 'C');
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int k        = *k_;
    int ldA      = *ldA_;
    int ldB      = *ldB_;
    double beta  = *beta_;
    int ldC      = *ldC_;

    dcomplex  alpha(alpha_[0], alpha_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (trans) ? k : n;
    int numRowsB = (trans) ? k : n;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (toupper(*trans_)!='N' && toupper(*trans_)!='C')
    {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (k<0) {
        info = 4;
    } else if (ldA<std::max(1,numRowsA)) {
        info = 7;
    } else if (ldB<std::max(1,numRowsB)) {
        info = 9;
    } else if (ldC<std::max(1,n)) {
        info = 12;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZHER2K", &info);
    }

//
//  Start the operations.
//
    if (!trans) {
        if (lower) {
            ulmBLAS::helr2k(n, k, alpha,
                            A, 1, ldA,
                            B, 1, ldB,
                            beta,
                            C, 1, ldC);
        } else {
            ulmBLAS::heur2k(n, k, alpha,
                            A, 1, ldA,
                            B, 1, ldB,
                            beta,
                            C, 1, ldC);
        }
    } else {
        alpha = ulmBLAS::conjugate(alpha);
        if (lower) {
            ulmBLAS::heur2k(n, k, alpha,
                            A, ldA, 1,
                            B, ldB, 1,
                            beta,
                            C, ldC, 1);
        } else {
            ulmBLAS::helr2k(n, k, alpha,
                            A, ldA, 1,
                            B, ldB, 1,
                            beta, C,
                            ldC, 1);
        }
    }
}


} // extern "C"
