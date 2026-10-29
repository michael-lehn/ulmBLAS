//
//  Numerical tests of the level 2 functions of the C interface.
//
#include <test/numeric/blas.h>
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

namespace {

using namespace test;

const CBLAS_TRANSPOSE  transposes[] = { CblasNoTrans, CblasTrans,
                                        CblasConjTrans };
const CBLAS_UPLO       upLos[]      = { CblasLower, CblasUpper };
const CBLAS_DIAG       diags[]      = { CblasNonUnit, CblasUnit };

struct Dims { int m, n; };

const Dims dims[] = { {0,3}, {3,0}, {1,1}, {2,5}, {7,3}, {17,19}, {40,33},
                      {130,67} };
const int numDims = sizeof(dims)/sizeof(dims[0]);

// pairs of increments (incX, incY), including negative increments
struct Incs { int x, y; };

const Incs incs[] = { {1,1}, {2,3}, {-1,1}, {1,-2}, {-3,-1} };
const int numIncs = sizeof(incs)/sizeof(incs[0]);

template <typename T>
void
fillNan(Vector<T> &x)
{
    for (int i=0; i<x.n; ++i) {
        x(i) = nan<T>();
    }
}

template <typename T>
bool
checkUnchanged(Context &ctx, const Vector<T> &x, const Vector<T> &x0,
               const std::string &what)
{
    return ctx.check(x.gapsUnchanged(x0), "%s: elements between the "
                     "increments modified", what.c_str());
}

//
//  GEMV: y = alpha*op(A)*x + beta*y
//
template <typename T>
bool
gemvCase(Context &ctx, Random &random, bool rowMajor, CBLAS_TRANSPOSE trans,
         int m, int n, int incX, int incY, T alpha, T beta)
{
    bool       t  = trans!=CblasNoTrans;
    Matrix<T>  A(m, n, rowMajor);
    Vector<T>  x(t ? m : n, incX), y(t ? n : m, incY);

    // As reference BLAS, return immediately if m or n is zero (y is not
    // scaled by beta in this case).
    bool quickReturn = m==0 || n==0;

    A.randomize(random);
    x.randomize(random);
    y.randomize(random);
    if (beta==T(0) && !quickReturn) {
        fillNan(y);
    }

    Result     ref = quickReturn
                   ? multiply(0, full(A, trans), fullVector(x), 1,
                              fullVector(y))
                   : multiply(toCdouble(alpha), full(A, trans), fullVector(x),
                              toCdouble(beta), fullVector(y));
    Vector<T>  y0 = y;

    blas::gemv(order(rowMajor), trans, m, n, alpha, A.ptr(), A.ld,
               x.ptr(), incX, beta, y.ptr(), incY);

    std::string what = format("gemv(%s,%c,m=%d,n=%d,incX=%d,incY=%d,"
                              "alpha=%s,beta=%s)",
                              orderName(order(rowMajor)), transChar(trans),
                              m, n, incX, incY, str(alpha).c_str(),
                              str(beta).c_str());
    return checkVector(ctx, y, ref, t ? m : n, what)
        && checkUnchanged(ctx, y, y0, what);
}

template <typename T>
void
testGemv(Context &ctx)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int o=0; o<2; ++o) {
        for (int t=0; t<3; ++t) {
            for (int d=0; d<numDims; ++d) {
                for (int i=0; i<numIncs; ++i) {
                    for (std::size_t a=0; a<s.size(); ++a) {
                        for (std::size_t b=0; b<s.size(); ++b) {
                            if (!gemvCase(ctx, random, o==1, transposes[t],
                                          dims[d].m, dims[d].n, incs[i].x,
                                          incs[i].y, s[a], s[b]))
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}

//
//  GER, GERU: A = alpha*x*y^T + A,  GERC: A = alpha*x*y^H + A
//
template <typename T>
bool
gerCase(Context &ctx, Random &random, bool conj, bool rowMajor, int m, int n,
        int incX, int incY, T alpha)
{
    Matrix<T>  A(m, n, rowMajor);
    Vector<T>  x(m, incX), y(n, incY);

    A.randomize(random);
    x.randomize(random);
    y.randomize(random);

    Full  yt(1, n);
    for (int j=0; j<n; ++j) {
        yt(0,j) = conj ? std::conj(toCdouble(y(j))) : toCdouble(y(j));
    }
    Result     ref = multiply(toCdouble(alpha), fullVector(x), yt, 1, full(A));
    Matrix<T>  A0  = A;

    if (conj) {
        blas::gerc(order(rowMajor), m, n, alpha, x.ptr(), incX, y.ptr(),
                   incY, A.ptr(), A.ld);
    } else {
        blas::geru(order(rowMajor), m, n, alpha, x.ptr(), incX, y.ptr(),
                   incY, A.ptr(), A.ld);
    }

    std::string what = format("%s(%s,m=%d,n=%d,incX=%d,incY=%d,alpha=%s)",
                              conj ? "gerc" : "geru",
                              orderName(order(rowMajor)), m, n, incX, incY,
                              str(alpha).c_str());
    return checkMatrix(ctx, A, ref, 1, 'A', what)
        && ctx.check(A.paddingUnchanged(A0), "%s: padding of A modified",
                     what.c_str());
}

template <typename T>
void
testGer(Context &ctx)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int c=0; c<(IsComplex<T>::value ? 2 : 1); ++c) {
        for (int o=0; o<2; ++o) {
            for (int d=0; d<numDims; ++d) {
                for (int i=0; i<numIncs; ++i) {
                    for (std::size_t a=0; a<s.size(); ++a) {
                        if (!gerCase(ctx, random, c==1, o==1, dims[d].m,
                                     dims[d].n, incs[i].x, incs[i].y, s[a]))
                        {
                            return;
                        }
                    }
                }
            }
        }
    }
}

//
//  SYMV, HEMV: y = alpha*A*x + beta*y with A symmetric/hermitian (only
//  triangle upLo referenced, imaginary parts of the diagonal assumed zero).
//
template <typename T>
bool
hemvCase(Context &ctx, Random &random, bool rowMajor, CBLAS_UPLO upLo, int n,
         int incX, int incY, T alpha, T beta)
{
    Matrix<T>  A(n, n, rowMajor);
    Vector<T>  x(n, incX), y(n, incY);

    A.randomize(random);
    x.randomize(random);
    y.randomize(random);
    poisonTriangle(A, upLo, false);
    if (IsComplex<T>::value) {
        for (int i=0; i<n; ++i) {
            A(i,i) = fromCdouble<T>(cdouble(std::real(A(i,i)), 77));
        }
    }
    if (beta==T(0)) {
        fillNan(y);
    }

    Result     ref = multiply(toCdouble(alpha),
                              fullSymmetric(A, upLo, IsComplex<T>::value),
                              fullVector(x), toCdouble(beta), fullVector(y));
    Vector<T>  y0 = y;

    blas::hemv(order(rowMajor), upLo, n, alpha, A.ptr(), A.ld, x.ptr(), incX,
               beta, y.ptr(), incY);

    std::string what = format("%s(%s,%s,n=%d,incX=%d,incY=%d,alpha=%s,"
                              "beta=%s)",
                              IsComplex<T>::value ? "hemv" : "symv",
                              orderName(order(rowMajor)),
                              upLo==CblasLower ? "L" : "U", n, incX, incY,
                              str(alpha).c_str(), str(beta).c_str());
    return checkVector(ctx, y, ref, n, what)
        && checkUnchanged(ctx, y, y0, what);
}

template <typename T>
void
testHemv(Context &ctx)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int o=0; o<2; ++o) {
        for (int u=0; u<2; ++u) {
            for (int d=0; d<numDims; ++d) {
                for (int i=0; i<numIncs; ++i) {
                    for (std::size_t a=0; a<s.size(); ++a) {
                        for (std::size_t b=0; b<s.size(); ++b) {
                            if (!hemvCase(ctx, random, o==1, upLos[u],
                                          dims[d].n, incs[i].x, incs[i].y,
                                          s[a], s[b]))
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}

//
//  SYR, HER:   A = alpha*x*x^T + A  (x^H, alpha real for HER)
//  SYR2, HER2: A = alpha*x*y^T + alpha*y*x^T + A
//              (HER2: alpha*x*y^H + conj(alpha)*y*x^H + A)
//
//  Only the triangle upLo of A is updated.
//
template <typename T>
bool
herCase(Context &ctx, Random &random, bool rank2, bool rowMajor,
        CBLAS_UPLO upLo, int n, int incX, int incY, T alpha)
{
    bool       herm = IsComplex<T>::value;
    Matrix<T>  A(n, n, rowMajor);
    Vector<T>  x(n, incX), y(n, incY);

    A.randomize(random);
    x.randomize(random);
    y.randomize(random);
    if (herm) {
        for (int i=0; i<n; ++i) {
            A(i,i) = fromCdouble<T>(cdouble(std::real(A(i,i)), 0));
        }
    }

    Full  fx = fullVector(x), fy = fullVector(y);
    Full  xh(1, n), yh(1, n);
    for (int j=0; j<n; ++j) {
        xh(0,j) = herm ? std::conj(fx(j,0)) : fx(j,0);
        yh(0,j) = herm ? std::conj(fy(j,0)) : fy(j,0);
    }
    cdouble  a   = toCdouble(alpha);
    Result   ref = multiply(a, fx, rank2 ? yh : xh, 1, full(A));
    if (rank2) {
        Result r2 = multiply(herm ? std::conj(a) : a, fy, xh, 0, full(A));
        for (std::size_t p=0; p<ref.value.a.size(); ++p) {
            ref.value.a[p] += r2.value.a[p];
            ref.bound.a[p] += r2.bound.a[p];
        }
    }

    Matrix<T>    A0 = A;
    CBLAS_ORDER  o  = order(rowMajor);

    if (rank2) {
        blas::her2(o, upLo, n, alpha, x.ptr(), incX, y.ptr(), incY,
                   A.ptr(), A.ld);
    } else {
        blas::her(o, upLo, n, realPart(alpha), x.ptr(), incX, A.ptr(), A.ld);
    }

    std::string what = format("%s(%s,%s,n=%d,incX=%d,incY=%d,alpha=%s)",
                              herm ? (rank2 ? "her2" : "her")
                                   : (rank2 ? "syr2" : "syr"),
                              orderName(o), upLo==CblasLower ? "L" : "U",
                              n, incX, incY, str(alpha).c_str());
    if (!checkMatrix(ctx, A, ref, 2, upLo==CblasLower ? 'L' : 'U', what)) {
        return false;
    }
    for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
            bool other = (upLo==CblasLower) ? i<j : i>j;
            if (other && !sameBits(A(i,j), A0(i,j))) {
                return ctx.check(false, "%s: element (%d,%d) outside of "
                                 "triangle modified", what.c_str(), i, j);
            }
        }
    }
    return ctx.check(A.paddingUnchanged(A0), "%s: padding of A modified",
                     what.c_str());
}

template <typename T>
void
testHer(Context &ctx, bool rank2)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int o=0; o<2; ++o) {
        for (int u=0; u<2; ++u) {
            for (int d=0; d<numDims; ++d) {
                for (int i=0; i<numIncs; ++i) {
                    for (std::size_t a=0; a<s.size(); ++a) {
                        T alpha = rank2 ? s[a] : T(realPart(s[a]));
                        if (!herCase(ctx, random, rank2, o==1, upLos[u],
                                     dims[d].n, incs[i].x, incs[i].y, alpha))
                        {
                            return;
                        }
                    }
                }
            }
        }
    }
}

//
//  TRMV: x = op(A)*x,  TRSV: solve op(A)*y = x, y overwrites x (checked by
//  the residual, see TRSM in level3.cc).
//
template <typename T>
bool
trmvCase(Context &ctx, Random &random, bool solve, bool rowMajor,
         CBLAS_UPLO upLo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
         int incX)
{
    Matrix<T>  A(n, n, rowMajor);
    Vector<T>  x(n, incX);

    A.randomize(random);
    if (solve) {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<n; ++i) {
                A(i,j) = (i==j) ? A(i,j) + T(std::real(A(i,j))>=0 ? 2 : -2)
                                : A(i,j) / T(n);
            }
        }
    }
    poisonTriangle(A, upLo, diag==CblasUnit);
    x.randomize(random);

    Full       opA = fullTriangular(A, upLo, diag, trans);
    Full       fx  = fullVector(x);
    Vector<T>  x0  = x;

    CBLAS_ORDER o = order(rowMajor);
    if (solve) {
        blas::trsv(o, upLo, trans, diag, n, A.ptr(), A.ld, x.ptr(), incX);
    } else {
        blas::trmv(o, upLo, trans, diag, n, A.ptr(), A.ld, x.ptr(), incX);
    }

    std::string what = format("%s(%s,%s,%c,%s,n=%d,incX=%d)",
                              solve ? "trsv" : "trmv", orderName(o),
                              upLo==CblasLower ? "L" : "U", transChar(trans),
                              diag==CblasUnit ? "U" : "N", n, incX);
    if (!solve) {
        Result ref = multiply(1, opA, fx, 0, fx);
        if (!checkVector(ctx, x, ref, n, what)) {
            return false;
        }
    } else {
        Result res = multiply(1, opA, fullVector(x), -1, fx);
        for (int i=0; i<n; ++i) {
            if (!isFinite(x(i))
             || std::abs(res.value(i,0))
                > tolerance<T>(n)*res.bound(i,0).real())
            {
                return ctx.check(false, "%s: residual %d is %g, bound %g, %s",
                                 what.c_str(), i, std::abs(res.value(i,0)),
                                 res.bound(i,0).real(), typeName<T>());
            }
        }
    }
    return checkUnchanged(ctx, x, x0, what);
}

template <typename T>
void
testTrmv(Context &ctx, bool solve)
{
    Random random;
    const int xincs[] = { 1, 2, -1, -3 };

    for (int o=0; o<2; ++o) {
        for (int u=0; u<2; ++u) {
            for (int t=0; t<3; ++t) {
                for (int dg=0; dg<2; ++dg) {
                    for (int d=0; d<numDims; ++d) {
                        for (int i=0; i<4; ++i) {
                            if (!trmvCase<T>(ctx, random, solve, o==1,
                                             upLos[u], transposes[t],
                                             diags[dg], dims[d].n, xincs[i]))
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}

//
//  Band and packed storage (BLAS and CBLAS conventions).  index(i,j) is the
//  position of a(i,j) in the array or -1 if the element is not stored:
//
//  - GeneralBand: m x n, kl subdiagonals and ku superdiagonals, leading
//    dimension ld >= kl+ku+1.  Column major: a(i,j) at (ku+i-j) + j*ld;
//    row major: at (kl+j-i) + i*ld.
//  - Band: n x n, triangle upLo with k off-diagonals, ld >= k+1.  Column
//    major: upper a(i,j) at (k+i-j) + j*ld, lower at (i-j) + j*ld; row
//    major: upper at (j-i) + i*ld, lower at (k+j-i) + i*ld.
//  - Packed: n x n, triangle upLo, column by column (column major) or row
//    by row (row major).
//
struct Storage
{
    enum Kind { GeneralBand, Band, Packed };

    Kind        kind;
    bool        rowMajor;
    CBLAS_UPLO  upLo;
    int         m, n, kl, ku, ld;

    long
    index(int i, int j) const
    {
        long N = n;
        if (kind==GeneralBand) {
            if (i-j>kl || j-i>ku) {
                return -1;
            }
            return rowMajor ? (kl+j-i) + long(i)*ld : (ku+i-j) + long(j)*ld;
        }
        bool upper = upLo==CblasUpper;
        if (upper ? i>j : i<j) {
            return -1;
        }
        if (kind==Band) {
            int k = upper ? ku : kl;
            if (std::abs(i-j)>k) {
                return -1;
            }
            if (!rowMajor) {
                return upper ? (k+i-j) + long(j)*ld : (i-j) + long(j)*ld;
            }
            return upper ? (j-i) + long(i)*ld : (k+j-i) + long(i)*ld;
        }
        if (!rowMajor) {
            return upper ? i + long(j)*(j+1)/2 : i + (2*N-j-1)*j/2;
        }
        return upper ? j + (2*N-i-1)*i/2 : j + long(i)*(i+1)/2;
    }

    // number of elements of the array (plus 4 elements that must not be
    // touched)
    long
    size() const
    {
        long s = (kind==Packed) ? long(n)*(n+1)/2
               : long(ld)*((kind==GeneralBand && rowMajor) ? m : n);
        return s + 4;
    }
};

template <typename T>
std::vector<T>
storageArray(const Storage &st)
{
    return std::vector<T>(std::size_t(st.size()), nan<T>());
}

//
//  The matrix with the elements of the array (zero where not stored) as a
//  column major Matrix for the helpers of reference.h.
//
template <typename T>
Matrix<T>
storedMatrix(const Storage &st, const std::vector<T> &a)
{
    Matrix<T> A(st.m, st.n, false, 0);
    for (int j=0; j<st.n; ++j) {
        for (int i=0; i<st.m; ++i) {
            long p = st.index(i, j);
            A(i,j) = (p>=0) ? a[p] : T(0);
        }
    }
    return A;
}

template <typename T>
void
randomizeStored(const Storage &st, std::vector<T> &a, Random &random)
{
    for (int j=0; j<st.n; ++j) {
        for (int i=0; i<st.m; ++i) {
            long p = st.index(i, j);
            if (p>=0) {
                a[p] = random.value<T>();
            }
        }
    }
}

template <typename T>
bool
sameArray(const std::vector<T> &a, const std::vector<T> &b)
{
    for (std::size_t p=0; p<a.size(); ++p) {
        if (!sameBits(a[p], b[p])) {
            return false;
        }
    }
    return true;
}

//
//  GBMV: y = alpha*op(A)*x + beta*y, A general band matrix
//
template <typename T>
bool
gbmvCase(Context &ctx, Random &random, bool rowMajor, CBLAS_TRANSPOSE trans,
         int m, int n, int kl, int ku, int incX, int incY, T alpha, T beta)
{
    bool     t  = trans!=CblasNoTrans;
    Storage  st = { Storage::GeneralBand, rowMajor, CblasUpper, m, n, kl, ku,
                    kl+ku+1+2 };
    std::vector<T> a = storageArray<T>(st);
    randomizeStored(st, a, random);

    Vector<T>  x(t ? m : n, incX), y(t ? n : m, incY);
    x.randomize(random);
    y.randomize(random);

    bool quickReturn = m==0 || n==0;
    if (beta==T(0) && !quickReturn) {
        fillNan(y);
    }
    Matrix<T>  A   = storedMatrix(st, a);
    Result     ref = quickReturn
                   ? multiply(0, full(A, trans), fullVector(x), 1,
                              fullVector(y))
                   : multiply(toCdouble(alpha), full(A, trans), fullVector(x),
                              toCdouble(beta), fullVector(y));
    Vector<T>       y0 = y;
    std::vector<T>  a0 = a;

    blas::gbmv(order(rowMajor), trans, m, n, kl, ku, alpha, &a[0], st.ld,
               x.ptr(), incX, beta, y.ptr(), incY);

    std::string what = format("gbmv(%s,%c,m=%d,n=%d,kl=%d,ku=%d,incX=%d,"
                              "incY=%d,alpha=%s,beta=%s), %s",
                              orderName(order(rowMajor)), transChar(trans),
                              m, n, kl, ku, incX, incY, str(alpha).c_str(),
                              str(beta).c_str(), typeName<T>());
    return checkVector(ctx, y, ref, t ? m : n, what)
        && checkUnchanged(ctx, y, y0, what)
        && ctx.check(sameArray(a, a0), "%s: A modified", what.c_str());
}

template <typename T>
void
testGbmv(Context &ctx)
{
    Random          random;
    std::vector<T>  s = scalars<T>();
    const int       kk[][2] = { {0,0}, {1,2}, {3,0}, {0,4}, {6,9} };

    for (int o=0; o<2; ++o) {
        for (int t=0; t<3; ++t) {
            for (int d=0; d<numDims-1; ++d) {
                for (int k=0; k<5; ++k) {
                    for (int i=0; i<numIncs; ++i) {
                        for (std::size_t a=0; a<s.size(); ++a) {
                            for (std::size_t b=0; b<s.size(); ++b) {
                                if (!gbmvCase(ctx, random, o==1,
                                              transposes[t], dims[d].m,
                                              dims[d].n, kk[k][0], kk[k][1],
                                              incs[i].x, incs[i].y, s[a],
                                              s[b]))
                                {
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

const int bandN[]    = { 0, 1, 2, 5, 17, 40 };
const int numBandN   = sizeof(bandN)/sizeof(bandN[0]);
const int bandK[]    = { 0, 1, 3, 45 };
const int numBandK   = sizeof(bandK)/sizeof(bandK[0]);

//
//  The symmetric or hermitian matrix (triangle upLo stored in band or packed
//  storage).  For hermitian matrices the imaginary parts of the stored
//  diagonal are set to 77: they must be ignored (assumed zero).
//
template <typename T>
void
markHermitianDiagonal(const Storage &st, std::vector<T> &a)
{
    if (IsComplex<T>::value) {
        for (int i=0; i<st.n; ++i) {
            long p = st.index(i, i);
            a[p] = fromCdouble<T>(cdouble(std::real(a[p]), 77));
        }
    }
}

//
//  SBMV, HBMV (band) and SPMV, HPMV (packed): y = alpha*A*x + beta*y
//
template <typename T>
bool
hbpmvCase(Context &ctx, Random &random, bool packed, bool rowMajor,
          CBLAS_UPLO upLo, int n, int k, int incX, int incY, T alpha, T beta)
{
    bool     herm = IsComplex<T>::value;
    Storage  st   = { packed ? Storage::Packed : Storage::Band, rowMajor,
                      upLo, n, n, k, k, k+1+2 };
    std::vector<T> a = storageArray<T>(st);
    randomizeStored(st, a, random);
    markHermitianDiagonal(st, a);

    Vector<T>  x(n, incX), y(n, incY);
    x.randomize(random);
    y.randomize(random);
    if (beta==T(0) && n>0) {
        fillNan(y);
    }
    Result     ref = multiply(toCdouble(alpha),
                              fullSymmetric(storedMatrix(st, a), upLo, herm),
                              fullVector(x), toCdouble(beta), fullVector(y));
    Vector<T>       y0 = y;
    std::vector<T>  a0 = a;

    CBLAS_ORDER o = order(rowMajor);
    if (packed) {
        blas::hpmv(o, upLo, n, alpha, &a[0], x.ptr(), incX, beta, y.ptr(),
                   incY);
    } else {
        blas::hbmv(o, upLo, n, k, alpha, &a[0], st.ld, x.ptr(), incX, beta,
                   y.ptr(), incY);
    }

    std::string what = format("%s(%s,%s,n=%d,k=%d,incX=%d,incY=%d,alpha=%s,"
                              "beta=%s), %s",
                              packed ? (herm ? "hpmv" : "spmv")
                                     : (herm ? "hbmv" : "sbmv"),
                              orderName(o), upLo==CblasLower ? "L" : "U",
                              n, packed ? -1 : k, incX, incY,
                              str(alpha).c_str(), str(beta).c_str(),
                              typeName<T>());
    return checkVector(ctx, y, ref, n, what)
        && checkUnchanged(ctx, y, y0, what)
        && ctx.check(sameArray(a, a0), "%s: A modified", what.c_str());
}

template <typename T>
void
testHbpmv(Context &ctx, bool packed)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int o=0; o<2; ++o) {
        for (int u=0; u<2; ++u) {
            for (int d=0; d<numBandN; ++d) {
                for (int k=0; k<(packed ? 1 : numBandK); ++k) {
                    for (int i=0; i<numIncs; ++i) {
                        for (std::size_t a=0; a<s.size(); ++a) {
                            for (std::size_t b=0; b<s.size(); ++b) {
                                if (!hbpmvCase(ctx, random, packed, o==1,
                                               upLos[u], bandN[d],
                                               bandK[k], incs[i].x,
                                               incs[i].y, s[a], s[b]))
                                {
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

//
//  TBMV, TPMV: x = op(A)*x,  TBSV, TPSV: solve op(A)*y = x (residual), A
//  triangular in band or packed storage.  For unit triangular matrices the
//  stored diagonal is NaN: it must not be referenced.
//
template <typename T>
bool
tbpCase(Context &ctx, Random &random, bool packed, bool solve, bool rowMajor,
        CBLAS_UPLO upLo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n, int k,
        int incX)
{
    Storage  st = { packed ? Storage::Packed : Storage::Band, rowMajor, upLo,
                    n, n, k, k, k+1+2 };
    std::vector<T> a = storageArray<T>(st);
    randomizeStored(st, a, random);
    for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
            long p = st.index(i, j);
            if (p<0) {
                continue;
            }
            if (i==j) {
                a[p] = (diag==CblasUnit) ? nan<T>()
                     : solve ? a[p] + T(std::real(a[p])>=0 ? 2 : -2)
                             : a[p];
            } else if (solve) {
                a[p] = a[p] / T(n);
            }
        }
    }

    Vector<T>  x(n, incX);
    x.randomize(random);

    Full            opA = fullTriangular(storedMatrix(st, a), upLo, diag,
                                         trans);
    Full            fx  = fullVector(x);
    Vector<T>       x0  = x;
    std::vector<T>  a0  = a;

    CBLAS_ORDER o = order(rowMajor);
    if (packed) {
        if (solve) {
            blas::tpsv(o, upLo, trans, diag, n, &a[0], x.ptr(), incX);
        } else {
            blas::tpmv(o, upLo, trans, diag, n, &a[0], x.ptr(), incX);
        }
    } else {
        if (solve) {
            blas::tbsv(o, upLo, trans, diag, n, k, &a[0], st.ld, x.ptr(),
                       incX);
        } else {
            blas::tbmv(o, upLo, trans, diag, n, k, &a[0], st.ld, x.ptr(),
                       incX);
        }
    }

    std::string what = format("%s(%s,%s,%c,%s,n=%d,k=%d,incX=%d), %s",
                              packed ? (solve ? "tpsv" : "tpmv")
                                     : (solve ? "tbsv" : "tbmv"),
                              orderName(o), upLo==CblasLower ? "L" : "U",
                              transChar(trans), diag==CblasUnit ? "U" : "N",
                              n, packed ? -1 : k, incX, typeName<T>());
    if (!solve) {
        Result ref = multiply(1, opA, fx, 0, fx);
        if (!checkVector(ctx, x, ref, n, what)) {
            return false;
        }
    } else {
        Result res = multiply(1, opA, fullVector(x), -1, fx);
        for (int i=0; i<n; ++i) {
            if (!isFinite(x(i))
             || std::abs(res.value(i,0))
                > tolerance<T>(n)*res.bound(i,0).real())
            {
                return ctx.check(false, "%s: residual %d is %g, bound %g",
                                 what.c_str(), i, std::abs(res.value(i,0)),
                                 res.bound(i,0).real());
            }
        }
    }
    return checkUnchanged(ctx, x, x0, what)
        && ctx.check(sameArray(a, a0), "%s: A modified", what.c_str());
}

template <typename T>
void
testTbp(Context &ctx, bool packed, bool solve)
{
    Random    random;
    const int xincs[] = { 1, 2, -1, -3 };

    for (int o=0; o<2; ++o) {
        for (int u=0; u<2; ++u) {
            for (int t=0; t<3; ++t) {
                for (int dg=0; dg<2; ++dg) {
                    for (int d=0; d<numBandN; ++d) {
                        for (int k=0; k<(packed ? 1 : numBandK); ++k) {
                            for (int i=0; i<4; ++i) {
                                if (!tbpCase<T>(ctx, random, packed, solve,
                                                o==1, upLos[u],
                                                transposes[t], diags[dg],
                                                bandN[d], bandK[k],
                                                xincs[i]))
                                {
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

//
//  SPR, HPR:   A = alpha*x*x^T + A  (x^H, alpha real for HPR)
//  SPR2, HPR2: A = alpha*x*y^T + alpha*y*x^T + A
//              (HPR2: alpha*x*y^H + conj(alpha)*y*x^H + A)
//
//  A in packed storage.  For hermitian matrices the imaginary parts of the
//  diagonal are 77 on entry (assumed zero) and must be zero on exit.  As in
//  the reference BLAS, nothing is changed for n = 0 or alpha = 0.
//
template <typename T>
bool
hprCase(Context &ctx, Random &random, bool rank2, bool rowMajor,
        CBLAS_UPLO upLo, int n, int incX, int incY, T alpha)
{
    bool     herm = IsComplex<T>::value;
    Storage  st   = { Storage::Packed, rowMajor, upLo, n, n, 0, 0, 0 };
    std::vector<T> a = storageArray<T>(st);
    randomizeStored(st, a, random);
    markHermitianDiagonal(st, a);

    Vector<T>  x(n, incX), y(n, incY);
    x.randomize(random);
    y.randomize(random);

    Full  fx = fullVector(x), fy = fullVector(y);
    Full  xh(1, n), yh(1, n);
    for (int j=0; j<n; ++j) {
        xh(0,j) = herm ? std::conj(fx(j,0)) : fx(j,0);
        yh(0,j) = herm ? std::conj(fy(j,0)) : fy(j,0);
    }
    Full     fa  = fullSymmetric(storedMatrix(st, a), upLo, herm);
    cdouble  ca  = toCdouble(alpha);
    Result   ref = multiply(ca, fx, rank2 ? yh : xh, 1, fa);
    if (rank2) {
        Result r2 = multiply(herm ? std::conj(ca) : ca, fy, xh, 0, fa);
        for (std::size_t p=0; p<ref.value.a.size(); ++p) {
            ref.value.a[p] += r2.value.a[p];
            ref.bound.a[p] += r2.bound.a[p];
        }
    }
    std::vector<T>  a0 = a;

    CBLAS_ORDER o = order(rowMajor);
    if (rank2) {
        blas::hpr2(o, upLo, n, alpha, x.ptr(), incX, y.ptr(), incY, &a[0]);
    } else {
        blas::hpr(o, upLo, n, realPart(alpha), x.ptr(), incX, &a[0]);
    }

    std::string what = format("%s(%s,%s,n=%d,incX=%d,incY=%d,alpha=%s), %s",
                              herm ? (rank2 ? "hpr2" : "hpr")
                                   : (rank2 ? "spr2" : "spr"),
                              orderName(o), upLo==CblasLower ? "L" : "U",
                              n, incX, incY, str(alpha).c_str(),
                              typeName<T>());
    if (n==0 || alpha==T(0)) {
        return ctx.check(sameArray(a, a0), "%s: A modified", what.c_str());
    }
    for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
            long p = st.index(i, j);
            if (p<0) {
                continue;
            }
            T expected = fromCdouble<T>(ref.value(i,j));
            if (!closeEnough(a[p], expected, ref.bound(i,j).real(), 2)
             || (herm && i==j && std::imag(a[p])!=0))
            {
                return ctx.check(false, "%s: element (%d,%d) is %s, "
                                 "expected %s", what.c_str(), i, j,
                                 str(a[p]).c_str(), str(expected).c_str());
            }
        }
    }
    for (long p=st.size()-4; p<st.size(); ++p) {
        if (!sameBits(a[p], a0[p])) {
            return ctx.check(false, "%s: element %ld after A modified",
                             what.c_str(), p);
        }
    }
    ctx.pass();
    return true;
}

template <typename T>
void
testHpr(Context &ctx, bool rank2)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int o=0; o<2; ++o) {
        for (int u=0; u<2; ++u) {
            for (int d=0; d<numBandN; ++d) {
                for (int i=0; i<numIncs; ++i) {
                    for (std::size_t a=0; a<s.size(); ++a) {
                        T alpha = rank2 ? s[a] : T(realPart(s[a]));
                        if (!hprCase(ctx, random, rank2, o==1, upLos[u],
                                     bandN[d], incs[i].x, incs[i].y, alpha))
                        {
                            return;
                        }
                    }
                }
            }
        }
    }
}

} // namespace

ULMBLAS_TEST(T0200, "level2/sgemv")  { testGemv<float>(ctx); }
ULMBLAS_TEST(T0201, "level2/dgemv")  { testGemv<double>(ctx); }
ULMBLAS_TEST(T0202, "level2/cgemv")  { testGemv<cfloat>(ctx); }
ULMBLAS_TEST(T0203, "level2/zgemv")  { testGemv<cdouble>(ctx); }

ULMBLAS_TEST(T0204, "level2/sger")   { testGer<float>(ctx); }
ULMBLAS_TEST(T0205, "level2/dger")   { testGer<double>(ctx); }
ULMBLAS_TEST(T0206, "level2/cgeru, cgerc")  { testGer<cfloat>(ctx); }
ULMBLAS_TEST(T0207, "level2/zgeru, zgerc")  { testGer<cdouble>(ctx); }

ULMBLAS_TEST(T0208, "level2/ssymv")  { testHemv<float>(ctx); }
ULMBLAS_TEST(T0209, "level2/dsymv")  { testHemv<double>(ctx); }
ULMBLAS_TEST(T0210, "level2/chemv")  { testHemv<cfloat>(ctx); }
ULMBLAS_TEST(T0211, "level2/zhemv")  { testHemv<cdouble>(ctx); }

ULMBLAS_TEST(T0212, "level2/ssyr")   { testHer<float>(ctx, false); }
ULMBLAS_TEST(T0213, "level2/dsyr")   { testHer<double>(ctx, false); }
ULMBLAS_TEST(T0214, "level2/cher")   { testHer<cfloat>(ctx, false); }
ULMBLAS_TEST(T0215, "level2/zher")   { testHer<cdouble>(ctx, false); }

ULMBLAS_TEST(T0216, "level2/ssyr2")  { testHer<float>(ctx, true); }
ULMBLAS_TEST(T0217, "level2/dsyr2")  { testHer<double>(ctx, true); }
ULMBLAS_TEST(T0218, "level2/cher2")  { testHer<cfloat>(ctx, true); }
ULMBLAS_TEST(T0219, "level2/zher2")  { testHer<cdouble>(ctx, true); }

ULMBLAS_TEST(T0220, "level2/strmv")  { testTrmv<float>(ctx, false); }
ULMBLAS_TEST(T0221, "level2/dtrmv")  { testTrmv<double>(ctx, false); }
ULMBLAS_TEST(T0222, "level2/ctrmv")  { testTrmv<cfloat>(ctx, false); }
ULMBLAS_TEST(T0223, "level2/ztrmv")  { testTrmv<cdouble>(ctx, false); }

ULMBLAS_TEST(T0224, "level2/strsv")  { testTrmv<float>(ctx, true); }
ULMBLAS_TEST(T0225, "level2/dtrsv")  { testTrmv<double>(ctx, true); }
ULMBLAS_TEST(T0226, "level2/ctrsv")  { testTrmv<cfloat>(ctx, true); }
ULMBLAS_TEST(T0227, "level2/ztrsv")  { testTrmv<cdouble>(ctx, true); }

ULMBLAS_TEST(T0228, "level2/sgbmv")   { testGbmv<float>(ctx); }
ULMBLAS_TEST(T0229, "level2/dgbmv")   { testGbmv<double>(ctx); }
ULMBLAS_TEST(T0230, "level2/cgbmv")   { testGbmv<cfloat>(ctx); }
ULMBLAS_TEST(T0231, "level2/zgbmv")   { testGbmv<cdouble>(ctx); }

ULMBLAS_TEST(T0232, "level2/ssbmv")   { testHbpmv<float>(ctx, false); }
ULMBLAS_TEST(T0233, "level2/dsbmv")   { testHbpmv<double>(ctx, false); }
ULMBLAS_TEST(T0234, "level2/chbmv")   { testHbpmv<cfloat>(ctx, false); }
ULMBLAS_TEST(T0235, "level2/zhbmv")   { testHbpmv<cdouble>(ctx, false); }

ULMBLAS_TEST(T0236, "level2/sspmv")   { testHbpmv<float>(ctx, true); }
ULMBLAS_TEST(T0237, "level2/dspmv")   { testHbpmv<double>(ctx, true); }
ULMBLAS_TEST(T0238, "level2/chpmv")   { testHbpmv<cfloat>(ctx, true); }
ULMBLAS_TEST(T0239, "level2/zhpmv")   { testHbpmv<cdouble>(ctx, true); }

ULMBLAS_TEST(T0240, "level2/stbmv")   { testTbp<float>(ctx, false, false); }
ULMBLAS_TEST(T0241, "level2/dtbmv")   { testTbp<double>(ctx, false, false); }
ULMBLAS_TEST(T0242, "level2/ctbmv")   { testTbp<cfloat>(ctx, false, false); }
ULMBLAS_TEST(T0243, "level2/ztbmv")   { testTbp<cdouble>(ctx, false, false); }

ULMBLAS_TEST(T0244, "level2/stbsv")   { testTbp<float>(ctx, false, true); }
ULMBLAS_TEST(T0245, "level2/dtbsv")   { testTbp<double>(ctx, false, true); }
ULMBLAS_TEST(T0246, "level2/ctbsv")   { testTbp<cfloat>(ctx, false, true); }
ULMBLAS_TEST(T0247, "level2/ztbsv")   { testTbp<cdouble>(ctx, false, true); }

ULMBLAS_TEST(T0248, "level2/stpmv")   { testTbp<float>(ctx, true, false); }
ULMBLAS_TEST(T0249, "level2/dtpmv")   { testTbp<double>(ctx, true, false); }
ULMBLAS_TEST(T0250, "level2/ctpmv")   { testTbp<cfloat>(ctx, true, false); }
ULMBLAS_TEST(T0251, "level2/ztpmv")   { testTbp<cdouble>(ctx, true, false); }

ULMBLAS_TEST(T0252, "level2/stpsv")   { testTbp<float>(ctx, true, true); }
ULMBLAS_TEST(T0253, "level2/dtpsv")   { testTbp<double>(ctx, true, true); }
ULMBLAS_TEST(T0254, "level2/ctpsv")   { testTbp<cfloat>(ctx, true, true); }
ULMBLAS_TEST(T0255, "level2/ztpsv")   { testTbp<cdouble>(ctx, true, true); }

ULMBLAS_TEST(T0256, "level2/sspr")    { testHpr<float>(ctx, false); }
ULMBLAS_TEST(T0257, "level2/dspr")    { testHpr<double>(ctx, false); }
ULMBLAS_TEST(T0258, "level2/chpr")    { testHpr<cfloat>(ctx, false); }
ULMBLAS_TEST(T0259, "level2/zhpr")    { testHpr<cdouble>(ctx, false); }

ULMBLAS_TEST(T0260, "level2/sspr2")   { testHpr<float>(ctx, true); }
ULMBLAS_TEST(T0261, "level2/dspr2")   { testHpr<double>(ctx, true); }
ULMBLAS_TEST(T0262, "level2/chpr2")   { testHpr<cfloat>(ctx, true); }
ULMBLAS_TEST(T0263, "level2/zhpr2")   { testHpr<cdouble>(ctx, true); }
