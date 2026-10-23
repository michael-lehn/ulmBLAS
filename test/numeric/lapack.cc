//
//  Tests of the C interface of the LAPACK functions (dgetrf, dlaswp).
//
//  The interface follows ATLAS: pivot indices are 0-based, for dlaswp k1 is
//  the first and k2-1 the last pivot applied.  The function names have the
//  prefix ULMLAPACK of the configuration.
//
//  For row major storage dgetrf follows ATLAS' clapack_dgetrf: the matrix is
//  factored as its transpose in column major storage, i.e. A*P = L*U with
//  column interchanges, L lower triangular and U unit upper triangular.  The
//  pivot indices refer to columns of A (bench/lutst.c, the ATLAS LU tester,
//  checks the same convention).
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
//  dgetrf for an m x n row major matrix with leading dimension ld >= n;
//  zeroRow >= 0 makes the matrix singular.
//
//  In memory, element (i,j) of A is A[i*ld+j].  Viewed as a column major
//  matrix with leading dimension ld the buffer is the n x m matrix A^T, and
//  the ATLAS convention A*P = L*U is P^T*A^T = U^T*L^T: the column major
//  factorization of A^T with row interchanges (unit lower U^T).  This is
//  checked with checkLu for the n x m column major view.
//
//  The matrix is placed between guard regions; the guards and the padding
//  (columns n, ..., ld-1 of each row) must not be modified.
//
bool
getrfRowMajorCase(Context &ctx, Random &random, int m, int n, int ld,
                  int zeroRow)
{
    const double         guardValue = -777;
    const std::size_t    guard      = 64 + 2*std::size_t(ld);
    const std::size_t    size       = std::size_t(ld)*std::max(1, m);

    std::vector<double>  buffer(guard + size + guard, guardValue);
    double               *A = &buffer[guard];

    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            A[i*ld+j] = (i==zeroRow) ? 0 : random.uniform();
        }
    }
    std::vector<double>  A0(A, A+size);
    std::vector<int>     piv(std::max(1, std::min(m, n)), -1);

    int info = ULMLAPACK(dgetrf)(CblasRowMajor, m, n, A, ld, &piv[0]);

    std::string what = format("dgetrf(row,m=%d,n=%d,ld=%d%s)", m, n, ld,
                              zeroRow>=0 ? ",singular" : "");

    for (std::size_t p=0; p<buffer.size(); ++p) {
        bool inMatrix = p>=guard && p<guard+size
                     && int((p-guard)%ld)<n && int((p-guard)/ld)<m;
        if (!inMatrix && buffer[p]!=guardValue) {
            return ctx.check(false, "%s: memory outside the matrix modified "
                             "(offset %ld)", what.c_str(),
                             long(p)-long(guard));
        }
    }

    std::vector<double> Af(A, A+size);
    return checkLu(ctx, n, m, A0, Af, ld, piv, info, what);
}

const int rowMajorSquare[]         = { 1, 2, 5, 33, 64, 65, 70, 200 };
const int rowMajorRectangular[][2] = { {1,7}, {7,1}, {3,5}, {5,3}, {4,6},
                                       {6,4}, {40,60}, {60,40}, {64,130},
                                       {130,64}, {97,131}, {131,97} };

// leading dimensions: exact (ld == n) and larger than necessary
const int ldExtra[] = { 0, 3, 17 };

void
testGetrfRowMajorSquare(Context &ctx)
{
    Random random;
    for (int i=0; i<8; ++i) {
        for (int l=0; l<3; ++l) {
            int n = rowMajorSquare[i];
            if (!getrfRowMajorCase(ctx, random, n, n, n+ldExtra[l], -1)) {
                return;
            }
        }
    }
}

void
testGetrfRowMajorRectangular(Context &ctx)
{
    Random random;
    for (int c=0; c<12; ++c) {
        for (int l=0; l<3; ++l) {
            int m = rowMajorRectangular[c][0], n = rowMajorRectangular[c][1];
            if (!getrfRowMajorCase(ctx, random, m, n, n+ldExtra[l], -1)) {
                return;
            }
        }
    }
}

void
testGetrfRowMajorSingular(Context &ctx)
{
    Random random;
    // a zero row k of A is a zero column of A^T: U^T(k,k) = 0, info = k+1,
    // also in the blocked algorithm (block size 64)
    const int cases[][3] = { {5,5,0}, {5,5,3}, {4,7,2}, {7,4,2},
                             {100,100,70}, {120,150,5}, {150,120,90} };
    for (int c=0; c<7; ++c) {
        if (!getrfRowMajorCase(ctx, random, cases[c][0], cases[c][1],
                               cases[c][1]+2, cases[c][2]))
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

ULMBLAS_TEST(T0401, "lapack/dgetrf row major, square")
{
    testGetrfRowMajorSquare(ctx);
}

ULMBLAS_TEST(T0402, "lapack/dgetrf singular")      { testGetrfSingular(ctx); }
ULMBLAS_TEST(T0403, "lapack/dlaswp")               { testLaswp(ctx); }

ULMBLAS_TEST(T0404, "lapack/dgetrf row major, m != n")
{
    testGetrfRowMajorRectangular(ctx);
}

ULMBLAS_TEST(T0405, "lapack/dgetrf row major, singular")
{
    testGetrfRowMajorSingular(ctx);
}
