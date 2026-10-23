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
