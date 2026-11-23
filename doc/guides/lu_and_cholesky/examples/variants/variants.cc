// All variants of chol.h, lu.h and lupiv.h, unblocked and blocked: the
// residual of the result and the speed.  Cholesky on a symmetric positive
// definite matrix, LU without pivoting on a diagonally dominant one, LU with
// pivoting on a random one.  The blocked variants use the BLAS of ulmBLAS
// (one thread) and the unblocked variant of the same number for the
// diagonal block or the panel.
#include "../common/util.h"
#include "chol.h"
#include "lu.h"
#include "lupiv.h"
#include <vector>

// ||A - L L^T|| / (n eps ||A||), lower triangles; A0 is overwritten
double
cholResidual(long n, double *A0, const double *L_)
{
    const double nrmA = normMax(n, n, A0, n);
    std::vector<double> L(n * n, 0.0);
    for (long j = 0; j < n; ++j) {
	for (long i = j; i < n; ++i) {
	    L[i + j * n] = L_[i + j * n];
	}
    }
    ulmBLAS::gemm(n, n, n, -1.0, false, L.data(), 1L, n, false, L.data(), n, 1L,
                  1.0, A0, 1L, n);
    double r = 0;
    for (long j = 0; j < n; ++j) {
	for (long i = j; i < n; ++i) {
	    r = std::max(r, std::abs(A0[i + j * n]));
	}
    }
    return r / (n * std::numeric_limits<double>::epsilon() * nrmA);
}

int
main()
{
    const long n = 1500, nb = 96;
    std::vector<double> A0(n * n), W(n * n), A(n * n);
    std::vector<long> piv(n);
    const double n3 = double(n) * n * n;
    std::printf("n = %ld, block size %ld, column major\n", n, nb);
    std::printf("%-23s %19s %19s\n", "", "unblocked", "blocked");
    std::printf("%-23s %9s %9s %9s %9s\n", "", "GFLOPS", "residual", "GFLOPS",
                "residual");

    // Cholesky
    spdMatrix(n, A0.data(), W.data());
    long (*cu[])(long, double *, long) = {chol::unb1, chol::unb2, chol::unb3};
    long (*cb[])(long, double *, long, long) = {chol::blk1, chol::blk2,
                                                chol::blk3};
    for (int v = 0; v < 3; ++v) {
	std::printf("Cholesky, variant %d    ", v + 1);
	for (int blocked = 0; blocked < 2; ++blocked) {
	    A = A0;
	    lap();
	    const long info =
	        blocked ? cb[v](n, A.data(), n, nb) : cu[v](n, A.data(), n);
	    const double t = lap();
	    W = A0;
	    std::printf(" %9.2f %9.2f", 1e-9 * n3 / 3 / t,
	                info ? NAN : cholResidual(n, W.data(), A.data()));
	}
	std::printf("\n");
    }

    // LU without pivoting
    randomMatrix(n, n, A0.data(), n);
    for (long i = 0; i < n; ++i) {
	A0[i + i * n] += double(n); // diagonally dominant
	piv[i] = i;
    }
    void (*lu_[])(long, double *, long) = {lu::unb1, lu::unb2, lu::unb3,
                                           lu::unb4, lu::unb5};
    void (*lb[])(long, double *, long, long) = {lu::blk1, lu::blk2, lu::blk3,
                                                lu::blk4, lu::blk5};
    for (int v = 0; v < 5; ++v) {
	std::printf("LU, variant %d          ", v + 1);
	for (int blocked = 0; blocked < 2; ++blocked) {
	    A = A0;
	    lap();
	    if (blocked) {
		lb[v](n, A.data(), n, nb);
	    } else {
		lu_[v](n, A.data(), n);
	    }
	    const double t = lap();
	    W = A0;
	    std::printf(" %9.2f %9.2f", 2e-9 * n3 / 3 / t,
	                luResidual(n, W.data(), A.data(), piv.data()));
	}
	std::printf("\n");
    }

    // LU with partial pivoting
    randomMatrix(n, n, A0.data(), n);
    long (*pu[])(long, long, double *, long,
                 long *) = {lupiv::unb3, lupiv::unb4, lupiv::unb5};
    long (*pb[])(long, double *, long, long *,
                 long) = {lupiv::blk3, lupiv::blk4, lupiv::blk5};
    for (int v = 0; v < 3; ++v) {
	std::printf("LU pivoting, variant %d ", v + 3);
	for (int blocked = 0; blocked < 2; ++blocked) {
	    A = A0;
	    lap();
	    const long info = blocked ? pb[v](n, A.data(), n, piv.data(), nb)
	                              : pu[v](n, n, A.data(), n, piv.data());
	    const double t = lap();
	    W = A0;
	    std::printf(" %9.2f %9.2f", 2e-9 * n3 / 3 / t,
	                info ? NAN
	                     : luResidual(n, W.data(), A.data(), piv.data()));
	}
	std::printf("\n");
    }
    std::printf("residual: ||A - L L^T||, ||A - LU|| or ||PA - LU||, "
                "divided by n eps ||A||\n");
}
