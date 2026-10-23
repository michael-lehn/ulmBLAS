//
//  Tests of the C interface of the LAPACK functions (dgetrf, dlaswp).
//
//  The interface follows ATLAS: pivot indices are 0-based, for dlaswp k1 is
//  the first and k2-1 the last pivot applied.  The function names have the
//  prefix ULMLAPACK of the configuration.
//
//  For row major storage the semantics of dgetrf is an open question (issue
//  #0004): row interchanges (P*A = L*U, as for column major storage) or
//  column interchanges (A = L*U*P with unit upper U, the factorization of
//  the transposed matrix, as ATLAS' clapack_dgetrf).
//
#include LAPACK_HEADER
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

namespace {

using namespace test;

//
//  Check P*A = L*U for the m x n column major matrix A0 and the result A
//  (L unit lower, U upper) and pivots of dgetrf.  Returns the expected info:
//  0, or the 1-based index of the first zero diagonal element of U.
//
bool
checkLu(Context &ctx, int m, int n, const std::vector<double> &A0,
        const std::vector<double> &A, int ld, const std::vector<int> &piv,
        int info, const std::string &what)
{
    int mn = std::min(m, n);

    // pivots: piv[i] in [i, m)
    for (int i=0; i<mn; ++i) {
        if (piv[i]<i || piv[i]>=m) {
            return ctx.check(false, "%s: invalid pivot piv[%d]=%d",
                             what.c_str(), i, piv[i]);
        }
    }

    // P*A0
    std::vector<double> PA(A0);
    for (int i=0; i<mn; ++i) {
        for (int j=0; j<n; ++j) {
            std::swap(PA[i+j*ld], PA[piv[i]+j*ld]);
        }
    }

    int expectedInfo = 0;
    for (int i=0; i<mn && expectedInfo==0; ++i) {
        if (A[i+i*ld]==0) {
            expectedInfo = i+1;
        }
    }
    if (!ctx.check(info==expectedInfo, "%s: info=%d, expected %d",
                   what.c_str(), info, expectedInfo))
    {
        return false;
    }

    for (int j=0; j<n; ++j) {
        for (int i=0; i<m; ++i) {
            double sum = 0, bound = 0;
            for (int l=0; l<=std::min(std::min(i, j), mn-1); ++l) {
                double lil = (l==i) ? 1 : A[i+l*ld];
                double ulj = A[l+j*ld];
                sum   += lil*ulj;
                bound += std::abs(lil*ulj);
            }
            if (!std::isfinite(A[i+j*ld])
             || std::abs(sum-PA[i+j*ld]) > tolerance<double>(mn)*bound
                                           + std::numeric_limits<double>::min())
            {
                return ctx.check(false, "%s: (P*A-L*U)(%d,%d) = %g, bound %g",
                                 what.c_str(), i, j, sum-PA[i+j*ld], bound);
            }
        }
    }
    ctx.pass();
    return true;
}

//
//  dgetrf for an m x n column major matrix; zeroColumn >= 0 makes the
//  matrix singular.
//
bool
getrfCase(Context &ctx, Random &random, int m, int n, int zeroColumn)
{
    int ld = std::max(1, m + 2);

    std::vector<double> A(std::size_t(ld)*std::max(1, n), -777);
    for (int j=0; j<n; ++j) {
        for (int i=0; i<m; ++i) {
            A[i+j*ld] = (j==zeroColumn) ? 0 : random.uniform();
        }
    }
    std::vector<double>  A0(A);
    std::vector<int>     piv(std::max(1, std::min(m, n)), -1);

    int info = ULMLAPACK(dgetrf)(CblasColMajor, m, n, &A[0], ld, &piv[0]);

    std::string what = format("dgetrf(col,m=%d,n=%d,ld=%d%s)", m, n, ld,
                              zeroColumn>=0 ? ",singular" : "");
    if (!checkLu(ctx, m, n, A0, A, ld, piv, info, what)) {
        return false;
    }
    for (int j=0; j<n; ++j) {
        for (int i=m; i<ld; ++i) {
            if (A[i+j*ld]!=-777) {
                return ctx.check(false, "%s: padding modified", what.c_str());
            }
        }
    }
    return true;
}

const int dims[][2] = { {0,0}, {0,3}, {3,0}, {1,1}, {5,5}, {7,4}, {4,7},
                        {33,33}, {70,70}, {131,97}, {97,131}, {200,200} };
const int numDims   = sizeof(dims)/sizeof(dims[0]);

void
testGetrf(Context &ctx)
{
    Random random;
    for (int d=0; d<numDims; ++d) {
        if (!getrfCase(ctx, random, dims[d][0], dims[d][1], -1)) {
            return;
        }
    }
}

void
testGetrfSingular(Context &ctx)
{
    Random random;
    // a zero column k makes U(k,k) zero (info = k+1), also in the blocked
    // algorithm (block size 64)
    const int cases[][3] = { {5,5,0}, {5,5,3}, {7,4,2}, {100,100,70},
                             {150,120,5} };
    for (int c=0; c<5; ++c) {
        if (!getrfCase(ctx, random, cases[c][0], cases[c][1], cases[c][2])) {
            return;
        }
    }
}

//
//  dgetrf for an m x n row major matrix.  The buffer has a large guard area
//  (writes beyond the matrix are detected, see issue #0003).  Returns which
//  factorization was computed:
//
//      "rows"     P*A = L*U (row interchanges)
//      "columns"  A = L*U*P (column interchanges, unit upper U)
//      "neither"  no valid factorization, or memory outside the matrix
//                 modified
//
std::string
getrfRowMajor(Random &random, int m, int n)
{
    int                  ld    = n + 2;
    std::size_t          size  = std::size_t(ld)*std::max(1, m);
    std::vector<double>  A(4*size + 4*std::size_t(ld)*std::max(m, n), -777);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            A[i*ld+j] = random.uniform();
        }
    }
    std::vector<double>  A0(A);
    std::vector<int>     piv(2*std::max(m, n) + 1, -1);

    int info = ULMLAPACK(dgetrf)(CblasRowMajor, m, n, &A[0], ld, &piv[0]);

    for (std::size_t p=0; p<A.size(); ++p) {
        bool inside = p<size && int(p%ld)<n;
        if (!inside && A[p]!=-777) {
            return "neither";
        }
    }

    // rows: the column major m x n matrix A^T^T, copy to column major
    std::vector<double> Ac(std::size_t(m)*n+1), A0c(Ac);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            Ac[i+j*m]  = A[i*ld+j];
            A0c[i+j*m] = A0[i*ld+j];
        }
    }
    Context c1, c2;
    if (checkLu(c1, m, n, A0c, Ac, std::max(1, m), piv, info, "")) {
        return "rows";
    }
    // columns: the buffer is the column major n x m matrix A^T
    if (checkLu(c2, n, m, A0, A, ld, piv, info, "")) {
        return "columns";
    }
    return "neither";
}

void
testGetrfRowMajorSquare(Context &ctx)
{
    Random       random;
    const int    sizes[] = { 1, 5, 33, 70, 200 };
    std::string  observed;

    for (int i=0; i<5; ++i) {
        std::string r = getrfRowMajor(random, sizes[i], sizes[i]);
        if (observed.empty()) {
            observed = r;
        } else if (observed!=r) {
            observed = "inconsistent";
        }
    }
    ctx.observed = observed;
    ctx.pass();
}

void
testGetrfRowMajorRectangular(Context &ctx)
{
    Random       random;
    const int    cases[][2] = { {6,4}, {4,6}, {131,97}, {97,131} };

    for (int c=0; c<4; ++c) {
        int m = cases[c][0], n = cases[c][1];
        std::string r = getrfRowMajor(random, m, n);
        if (!ctx.check(r!="neither", "dgetrf(row,m=%d,n=%d): no valid "
                       "factorization or memory outside the matrix modified",
                       m, n))
        {
            return;
        }
    }
}

//
//  dlaswp: row interchanges A(i,:) <-> A(piv[i],:) for i = k1, ..., k2-1
//  (column major).  The result must be exact.
//
void
testLaswp(Context &ctx)
{
    Random random;
    const int cases[][4] = { {5,3,0,5}, {10,1,2,7}, {40,33,0,40},
                             {64,70,10,50}, {9,0,0,9}, {6,4,3,3} };

    for (int c=0; c<6; ++c) {
        int m = cases[c][0], n = cases[c][1], k1 = cases[c][2],
            k2 = cases[c][3], ld = m+1;

        std::vector<double> A(std::size_t(ld)*std::max(1, n));
        for (std::size_t p=0; p<A.size(); ++p) {
            A[p] = random.uniform();
        }
        std::vector<int> piv(m);
        for (int i=0; i<m; ++i) {
            piv[i] = i + int((random.uniform()+1)/2*(m-i)) % (m-i);
        }

        std::vector<double> B(A);
        for (int i=k1; i<k2; ++i) {
            for (int j=0; j<n; ++j) {
                std::swap(B[i+j*ld], B[piv[i]+j*ld]);
            }
        }

        ULMLAPACK(dlaswp)(n, &A[0], ld, k1, k2, &piv[0], 1);

        if (!ctx.check(A==B, "dlaswp(m=%d,n=%d,k1=%d,k2=%d): wrong result",
                       m, n, k1, k2))
        {
            return;
        }
    }
}

} // namespace

ULMBLAS_TEST(T0400, "lapack/dgetrf column major")  { testGetrf(ctx); }

ULMBLAS_TEST_DESIGN(T0401, "lapack/dgetrf row major, square", 4, "rows")
{
    testGetrfRowMajorSquare(ctx);
}

ULMBLAS_TEST(T0402, "lapack/dgetrf singular")      { testGetrfSingular(ctx); }
ULMBLAS_TEST(T0403, "lapack/dlaswp")               { testLaswp(ctx); }

ULMBLAS_TEST_XFAIL(T0404, "lapack/dgetrf row major, m != n", 3)
{
    testGetrfRowMajorRectangular(ctx);
}
