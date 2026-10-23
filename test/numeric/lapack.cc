//
//  Tests of the C interface of the LAPACK functions (dgetrf, dlaswp).
//
//  The interface follows ATLAS' C interface (clapack_dgetrf): pivot indices
//  are 0-based (ATLAS converts to 1-based indices only in its Fortran 77
//  wrappers), the return value is LAPACK's 1-based INFO.  For dlaswp k1 is
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
//  Pivot indices
//
//  checkPivots: dgetrf on the column major mx x nx view (leading dimension
//  ld) of buffer A must return exactly the pivot indices expected.  Entries
//  of piv beyond min(mx,nx) must not be written.
//
bool
checkPivots(Context &ctx, const std::vector<int> &expected,
            const std::vector<int> &piv, int info, const std::string &what)
{
    for (std::size_t i=0; i<expected.size(); ++i) {
        if (piv[i]!=expected[i]) {
            return ctx.check(false, "%s: piv[%d] = %d, expected %d (0-based)",
                             what.c_str(), int(i), piv[i], expected[i]);
        }
    }
    for (std::size_t i=expected.size(); i<piv.size(); ++i) {
        if (piv[i]!=-1) {
            return ctx.check(false, "%s: piv[%d] written beyond min(m,n)",
                             what.c_str(), int(i));
        }
    }
    return ctx.check(info==0, "%s: info = %d, expected 0", what.c_str(), info);
}

//
//  Hand computed examples.  The same 3 x 3 data
//
//          1  2  9
//          4  5  6
//          7  8  3
//
//  is factored as column major matrix (row interchanges, P*A = L*U):
//  pivots 2, 2, 2, and as row major matrix (column interchanges, ATLAS
//  convention A*P = L*U): pivots 2, 1, 2 (largest entries 9 of row 0, then
//  11/3 of the updated row 1).
//
bool
pivotExample(Context &ctx, bool rowMajor)
{
    const double data[3][3] = { {1, 2, 9}, {4, 5, 6}, {7, 8, 3} };
    std::vector<double> A(9);
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            if (rowMajor) {
                A[i*3+j] = data[i][j];
            } else {
                A[i+j*3] = data[i][j];
            }
        }
    }
    std::vector<int> piv(4, -1);
    int info = ULMLAPACK(dgetrf)(rowMajor ? CblasRowMajor : CblasColMajor,
                                 3, 3, &A[0], 3, &piv[0]);
    std::vector<int> expected(3, 2);
    if (rowMajor) {
        expected[1] = 1;
    }
    return checkPivots(ctx, expected, piv, info,
                       format("dgetrf(%s, 3x3 example)",
                              rowMajor ? "row" : "col"));
}

//
//  Matrices with prescribed pivots: B = P^T*L*U with L unit lower
//  triangular, |L(i,j)| <= 1/2 below the diagonal, and U upper triangular
//  with |U(j,j)| >= 1.  Partial pivoting then selects exactly the
//  prescribed pivots: in each step the candidate on the prescribed row is
//  |U(j,j)|, all others are at most |U(j,j)|/2.  The prescribed pivots
//  contain interchanges and non-interchanges.
//
//  For column major storage B is the m x n matrix A; for row major storage
//  B is the transposed n x m matrix A^T (pivots refer to columns of A).
//
bool
prescribedPivotsCase(Context &ctx, Random &random, bool rowMajor, int m,
                     int n, int ld)
{
    int mx = rowMajor ? n : m;          // dimensions of B (column major view)
    int nx = rowMajor ? m : n;
    int mn = std::min(mx, nx);

    std::vector<int> expected(mn);
    for (int j=0; j<mn; ++j) {
        int r = int((random.uniform()+1)/2*(mx-j)) % std::max(1, mx-j);
        expected[j] = (j%3==0) ? mx-1 : (j%3==1) ? j : j + r;
    }

    std::vector<double> B(std::size_t(ld)*std::max(1, nx), -777);

    // B = L*U, then the interchanges in reverse order: B = P^T*L*U
    std::vector<double> L(std::size_t(mx)*mn), U(std::size_t(mn)*nx);
    for (int l=0; l<mn; ++l) {
        for (int i=0; i<mx; ++i) {
            L[i+l*mx] = (i==l) ? 1 : (i>l) ? 0.5*random.uniform() : 0;
        }
        for (int j=0; j<nx; ++j) {
            U[l+j*mn] = (j==l) ? (random.uniform()<0 ? -1.5 : 1.5)
                                 + 0.5*random.uniform()
                      : (j>l) ? random.uniform() : 0;
        }
    }
    for (int j=0; j<nx; ++j) {
        for (int i=0; i<mx; ++i) {
            double sum = 0;
            for (int l=0; l<mn; ++l) {
                sum += L[i+l*mx]*U[l+j*mn];
            }
            B[i+j*ld] = sum;
        }
    }
    for (int j=mn-1; j>=0; --j) {
        for (int k=0; k<nx; ++k) {
            std::swap(B[j+k*ld], B[expected[j]+k*ld]);
        }
    }

    std::vector<double>  B0(B);
    std::vector<int>     piv(mn+3, -1);

    int info = ULMLAPACK(dgetrf)(rowMajor ? CblasRowMajor : CblasColMajor,
                                 m, n, &B[0], ld, &piv[0]);

    std::string what = format("dgetrf(%s,m=%d,n=%d,ld=%d, prescribed "
                              "pivots)", rowMajor ? "row" : "col", m, n, ld);
    std::vector<int> pivMn(piv.begin(), piv.begin()+mn);
    return checkPivots(ctx, expected, piv, info, what)
        && checkLu(ctx, mx, nx, B0, B, ld, pivMn, info, what);
}

void
testPivots(Context &ctx, bool rowMajor)
{
    Random random(11);
    if (!pivotExample(ctx, rowMajor)) {
        return;
    }
    // m, n of A and extra leading dimension
    const int cases[][3] = { {1,1,0}, {2,2,0}, {5,5,2}, {7,4,0}, {4,7,3},
                             {33,33,1}, {64,64,0}, {65,65,4}, {150,150,3},
                             {150,97,0}, {97,150,5}, {200,70,2},
                             {70,200,0} };
    for (int c=0; c<13; ++c) {
        int m = cases[c][0], n = cases[c][1];
        int ld = (rowMajor ? n : m) + cases[c][2];
        if (!prescribedPivotsCase(ctx, random, rowMajor, m, n, ld)) {
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

ULMBLAS_TEST(T0406, "lapack/dgetrf pivot indices, column major")
{
    testPivots(ctx, false);
}

ULMBLAS_TEST(T0407, "lapack/dgetrf pivot indices, row major")
{
    testPivots(ctx, true);
}
