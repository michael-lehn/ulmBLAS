//
//  ?gemmtr (BLAS of LAPACK 3.12) through the C interface: the lower or
//  upper triangle of the n x n matrix C is set to alpha*op(A)*op(B) +
//  beta*C.  The result is compared with the triangle of the full product
//  (reference.h).  The other triangle of C holds NaN on entry and must be
//  unchanged on return: it is neither written nor read (a read would turn
//  the triangle into NaN).  For beta = 0, C holds NaN everywhere.
//
#include <cmath>
#include <cstring>
#include <test/numeric/blas.h>
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

namespace {

using namespace test;

const CBLAS_TRANSPOSE transposes[] = {CblasNoTrans, CblasTrans, CblasConjTrans};
const CBLAS_UPLO upLos[] = {CblasLower, CblasUpper};

template <typename T>
bool
gemmtrCase(Context &ctx, Random &random, bool rowMajor, CBLAS_UPLO upLo,
           CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int n, int k,
           T alpha, T beta)
{
    bool tA = transA != CblasNoTrans, tB = transB != CblasNoTrans;
    Matrix<T> A(tA ? k : n, tA ? n : k, rowMajor);
    Matrix<T> B(tB ? n : k, tB ? k : n, rowMajor, 5);
    Matrix<T> C(n, n, rowMajor, 4);

    A.randomize(random);
    B.randomize(random);
    C.randomize(random);
    for (int j = 0; j < n; ++j) {
	for (int i = 0; i < n; ++i) {
	    bool other = (upLo == CblasLower) ? i < j : i > j;
	    if (other || beta == T(0)) {
		C(i, j) = nan<T>();
	    }
	}
    }

    Result ref = multiply(toCdouble(alpha), full(A, transA), full(B, transB),
                          toCdouble(beta), full(C));
    Matrix<T> C0 = C;

    blas::gemmtr(order(rowMajor), upLo, transA, transB, n, k, alpha, A.ptr(),
                 A.ld, B.ptr(), B.ld, beta, C.ptr(), C.ld);

    std::string what =
        format("gemmtr(%s,%c,%c,%c,n=%d,k=%d,alpha=%s,beta=%s)",
	       orderName(order(rowMajor)), upLo == CblasLower ? 'L' : 'U',
	       transChar(transA), transChar(transB), n, k, str(alpha).c_str(),
	       str(beta).c_str());

    //  the other triangle: bitwise unchanged
    for (int j = 0; j < n; ++j) {
	for (int i = 0; i < n; ++i) {
	    bool other = (upLo == CblasLower) ? i < j : i > j;
	    if (other && std::memcmp(&C(i, j), &C0(i, j), sizeof(T)) != 0) {
		return ctx.check(false,
		                 "%s: C(%d,%d) outside the triangle "
		                 "modified",
		                 what.c_str(), i, j);
	    }
	}
    }
    return checkMatrix(ctx, C, ref, k, upLo == CblasLower ? 'L' : 'U', what) &&
           ctx.check(C.paddingUnchanged(C0), "%s: padding of C modified",
                     what.c_str());
}

struct Dims
{
    int n, k;
};

//
//  All transpositions, both triangles and storage orders, alpha and beta
//  from scalars() (zero, one, other), dimensions within one block column
//  of gemmtr (NB = 64) and across several
//
template <typename T>
void
testGemmtr(Context &ctx, const Dims *dims, int numDims, bool allScalars)
{
    Random random;
    std::vector<T> s = scalars<T>();
    if (!allScalars) {
	s = {fromCdouble<T>(cdouble(-0.75, 0.375))};
    }

    for (int o = 0; o < 2; ++o) {
	for (CBLAS_UPLO upLo : upLos) {
	    for (CBLAS_TRANSPOSE ta : transposes) {
		for (CBLAS_TRANSPOSE tb : transposes) {
		    for (int d = 0; d < numDims; ++d) {
			for (const T &a : s) {
			    for (const T &b : s) {
				if (!gemmtrCase(ctx, random, o == 1, upLo, ta,
				                tb, dims[d].n, dims[d].k, a,
				                b)) {
				    return;
				}
			    }
			}
		    }
		}
	    }
	}
    }
}

const Dims small[] = {{0, 2}, {3, 0}, {1, 1}, {2, 3}, {5, 7}, {17, 19}};
const Dims blocks[] = {{63, 5}, {64, 3}, {65, 9}, {130, 70}};

} // namespace

ULMBLAS_TEST(T0360, "level3/?gemmtr: triangles of small products, all "
                    "transpositions and storage orders, alpha and beta "
                    "0, 1, other")
{
    testGemmtr<float>(ctx, small, 6, true);
    testGemmtr<double>(ctx, small, 6, true);
    testGemmtr<cfloat>(ctx, small, 6, true);
    testGemmtr<cdouble>(ctx, small, 6, true);
}

ULMBLAS_TEST(T0361, "level3/?gemmtr: n = 63 ... 130 (one and several block "
                    "columns)")
{
    testGemmtr<float>(ctx, blocks, 4, false);
    testGemmtr<double>(ctx, blocks, 4, false);
    testGemmtr<cfloat>(ctx, blocks, 4, false);
    testGemmtr<cdouble>(ctx, blocks, 4, false);
}
