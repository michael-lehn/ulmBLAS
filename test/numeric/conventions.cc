//
//  Tests of conventions that the BLAS documentation leaves open.
//
#include <cmath>
#include <limits>
#include <type_traits>
#include <test/numeric/blas.h>
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

namespace {

using namespace test;

template <typename T>
bool
hasNan(const std::vector<T> &v)
{
    for (std::size_t i=0; i<v.size(); ++i) {
        if (!isFinite(v[i])) {
            return true;
        }
    }
    return false;
}

//
//  Calls each function with alpha=0, beta=1 (where applicable) and NaN in
//  the operands multiplied by alpha (A, B, x, y), for both storage orders and
//  both triangles.  As reference BLAS, ulmBLAS must not reference these
//  operands.  Returns the names of the functions that propagate the NaN into
//  the result ("none" if there are none).
//
template <typename T>
std::string
nanPropagation(const char *prefix, CBLAS_ORDER o, CBLAS_UPLO u)
{
    const int    n   = 4;
    const T      nan = test::nan<T>();
    const T      one(1);
    std::vector<T>  bad(n*n, nan);
    std::vector<T>  out;
    std::string     result;
    bool            herm = IsComplex<T>::value;
    CBLAS_SIDE      s = CblasLeft;
    CBLAS_TRANSPOSE t = CblasNoTrans;

#define ULMBLAS_RECORD(NAME, CALL)                                          \
    out.assign(n*n, one);                                                   \
    CALL;                                                                   \
    if (hasNan(out)) {                                                      \
        result += std::string(result.empty() ? "" : " ") + prefix + (NAME); \
    }

    ULMBLAS_RECORD("gemv", blas::gemv(o, t, n, n, T(0), &bad[0], n, &bad[0],
                                      1, one, &out[0], 1));
    ULMBLAS_RECORD(herm ? "geru" : "ger",
                   blas::geru(o, n, n, T(0), &bad[0], 1, &bad[0], 1,
                              &out[0], n));
    if (herm) {
        ULMBLAS_RECORD("gerc", blas::gerc(o, n, n, T(0), &bad[0], 1,
                                          &bad[0], 1, &out[0], n));
    }
    ULMBLAS_RECORD(herm ? "hemv" : "symv",
                   blas::hemv(o, u, n, T(0), &bad[0], n, &bad[0], 1, one,
                              &out[0], 1));
    ULMBLAS_RECORD(herm ? "her" : "syr",
                   blas::her(o, u, n, realPart(T(0)), &bad[0], 1, &out[0],
                             n));
    ULMBLAS_RECORD(herm ? "her2" : "syr2",
                   blas::her2(o, u, n, T(0), &bad[0], 1, &bad[0], 1,
                              &out[0], n));
    ULMBLAS_RECORD(herm ? "hpr" : "spr",
                   blas::hpr(o, u, n, realPart(T(0)), &bad[0], 1, &out[0]));
    ULMBLAS_RECORD(herm ? "hpr2" : "spr2",
                   blas::hpr2(o, u, n, T(0), &bad[0], 1, &bad[0], 1,
                              &out[0]));
    ULMBLAS_RECORD("gemm", blas::gemm(o, t, t, n, n, n, T(0), &bad[0], n,
                                      &bad[0], n, one, &out[0], n));
    ULMBLAS_RECORD("symm", blas::symm(o, s, u, n, n, T(0), &bad[0], n,
                                      &bad[0], n, one, &out[0], n));
    ULMBLAS_RECORD("syrk", blas::syrk(o, u, t, n, n, T(0), &bad[0], n, one,
                                      &out[0], n));
    ULMBLAS_RECORD("syr2k", blas::syr2k(o, u, t, n, n, T(0), &bad[0], n,
                                        &bad[0], n, one, &out[0], n));
    if (herm) {
        ULMBLAS_RECORD("hemm", blas::hemm(o, s, u, n, n, T(0), &bad[0], n,
                                          &bad[0], n, one, &out[0], n));
        ULMBLAS_RECORD("herk", blas::herk(o, u, t, n, n, realPart(T(0)),
                                          &bad[0], n, realPart(one),
                                          &out[0], n));
        ULMBLAS_RECORD("her2k", blas::her2k(o, u, t, n, n, T(0), &bad[0], n,
                                            &bad[0], n, realPart(one),
                                            &out[0], n));
    }
#undef ULMBLAS_RECORD

    return result.empty() ? std::string("none") : result;
}

//
//  With alpha=0, symv and hemv only compute y = beta*y, also for beta=0 (y
//  need not be set) and with NaN in A and x.
//
template <typename T>
bool
alphaZeroScalesY()
{
    const int    n   = 4;
    const T      nan = test::nan<T>();
    std::vector<T>  bad(n*n, nan);
    std::vector<T>  y(n);

    for (int i=0; i<n; ++i) {
        y[i] = T(i+1);
    }
    blas::hemv(CblasColMajor, CblasUpper, n, T(0), &bad[0], n, &bad[0], 1,
               T(0.5), &y[0], 1);
    for (int i=0; i<n; ++i) {
        if (!(y[i]==T(0.5)*T(i+1))) {
            return false;
        }
    }
    std::vector<T>  z(n, nan);
    blas::hemv(CblasColMajor, CblasLower, n, T(0), &bad[0], n, &bad[0], 1,
               T(0), &z[0], 1);
    for (int i=0; i<n; ++i) {
        if (!(z[i]==T(0))) {
            return false;
        }
    }
    return true;
}

//
//  0-based indices returned by i?amax for vectors with NaN, Inf and values
//  whose |re| + |im| overflows.  For complex types the real parts are
//  given; the imaginary parts are zero, except in the last vectors (complex
//  only).  As in Reference-LAPACK after 3.12.1: the first NaN, otherwise
//  the first maximum.
//
template <typename T>
std::string
iamaxWithNan(const char *prefix)
{
    typedef typename Real<T>::type R;

    const R nan = std::numeric_limits<R>::quiet_NaN();
    const R inf = std::numeric_limits<R>::infinity();
    const R v1[] = { 1, nan, 5, -7, 2 };
    const R v2[] = { nan, 2, 3 };
    const R v3[] = { 2, inf, nan, 9 };
    const R v4[] = { 2, -inf, inf, 9 };
    const R v5[] = { -3, 7, -7, 1 };
    const R *v[5] = { v1, v2, v3, v4, v5 };
    const int n[5] = { 5, 3, 4, 4, 4 };

    std::string result = std::string(prefix) + ":";
    for (int k=0; k<5; ++k) {
        std::vector<T> x(n[k]);
        for (int i=0; i<n[k]; ++i) {
            x[i] = T(v[k][i]);
        }
        result += format(" %d", blas::iamax(n[k], &x[0], 1));
    }
    return result;
}

template <typename R>
std::string
iamaxComplex(const char *prefix)
{
    typedef std::complex<R> T;

    const R nan = std::numeric_limits<R>::quiet_NaN();
    const R inf = std::numeric_limits<R>::infinity();
    const R big = std::numeric_limits<R>::max();
    // NaN in the imaginary part; Inf in the imaginary part before a NaN;
    // |re| + |im| overflows for the last three elements
    const T v1[] = { T(1, 0), T(2, nan), T(nan, 0) };
    const T v2[] = { T(1, 0), T(0, -inf), T(5, 0), T(0, nan) };
    const T v3[] = { T(1, 0), T(big, big/2), T(big, big), T(big/2, big) };
    const T *v[3] = { v1, v2, v3 };
    const int n[3] = { 3, 4, 4 };

    std::string result = std::string(prefix) + ":";
    for (int k=0; k<3; ++k) {
        result += format(" %d", blas::iamax(n[k], v[k], 1));
    }
    return result;
}

} // namespace

ULMBLAS_TEST(T0600, "conventions/alpha=0 with NaN in unreferenced operands")
{
    const CBLAS_ORDER order[2] = { CblasColMajor, CblasRowMajor };
    const CBLAS_UPLO  uplo[2]  = { CblasLower, CblasUpper };

    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            std::string observed
                = nanPropagation<float>("s", order[i], uplo[j]) + "; "
                + nanPropagation<double>("d", order[i], uplo[j]) + "; "
                + nanPropagation<cfloat>("c", order[i], uplo[j]) + "; "
                + nanPropagation<cdouble>("z", order[i], uplo[j]);
            ctx.check(observed=="none; none; none; none",
                      "order %d, uplo %d: NaN propagated by %s",
                      i, j, observed.c_str());
        }
    }
    ctx.check(alphaZeroScalesY<float>(), "ssymv: y != beta*y");
    ctx.check(alphaZeroScalesY<double>(), "dsymv: y != beta*y");
    ctx.check(alphaZeroScalesY<cfloat>(), "chemv: y != beta*y");
    ctx.check(alphaZeroScalesY<cdouble>(), "zhemv: y != beta*y");
}

ULMBLAS_TEST(T0601, "conventions/i?amax with NaN and Inf")
{
    std::string observed = iamaxWithNan<float>("s") + "; "
                         + iamaxWithNan<double>("d") + "; "
                         + iamaxWithNan<cfloat>("c") + "; "
                         + iamaxWithNan<cdouble>("z");
    ctx.check(observed=="s: 1 0 2 1 1; d: 1 0 2 1 1; "
                        "c: 1 0 2 1 1; z: 1 0 2 1 1",
              "%s", observed.c_str());
    observed = iamaxComplex<float>("c") + "; " + iamaxComplex<double>("z");
    ctx.check(observed=="c: 1 3 2; z: 1 3 2", "%s", observed.c_str());
}

//
//  Results of calls for which the reference BLAS skips a zero element of x,
//  y, A or B (Fortran: IF (X(J).NE.ZERO) ...), so that NaN or Inf in the
//  corresponding column of A is not propagated, or multiplies by a
//  structural zero.  ulmBLAS computes these products (issue #0024).  Each
//  entry is name=result; the reference BLAS gives the value in brackets.
//
namespace {

using namespace test;

std::string
num(double x)
{
    return std::isnan(x) ? std::string("nan") : format("%g", x);
}

std::string
zeroSkips()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::string  r;

    {   // dgemv N: A = [NaN], x = [0], beta = 0 (ref: NaN)
        double A[1] = { nan }, x[1] = { 0 }, y[1] = { 7 };
        cblas_dgemv(CblasColMajor, CblasNoTrans, 1, 1, 1, A, 1, x, 1, 0, y, 1);
        r += "gemv=" + num(y[0]);
    }
    {   // dger: x = [NaN], y = [0], A = [1] (ref: 1)
        double A[1] = { 1 }, x[1] = { nan }, y[1] = { 0 };
        cblas_dger(CblasColMajor, 1, 1, 1, x, 1, y, 1, A, 1);
        r += " ger=" + num(A[0]);
    }
    {   // dsyr L: x = (0, NaN), A21 = 2 (ref: 2)
        double A[4] = { 1, 2, -1, 3 }, x[2] = { 0, nan };
        cblas_dsyr(CblasColMajor, CblasLower, 2, 1, x, 1, A, 2);
        r += " syr=" + num(A[1]);
    }
    {   // dspr L: alpha = Inf, x = (0, 1), AP = (1, 2, 3) (ref: AP(1) = 1)
        double AP[3] = { 1, 2, 3 }, x[2] = { 0, 1 };
        cblas_dspr(CblasColMajor, CblasLower, 2, inf, x, 1, AP);
        r += " spr=" + num(AP[0]);
    }
    {   // dtrmv U N: A = [1 NaN; 0 1], x = (1, 0) (ref: x1 = 1)
        double A[4] = { 1, 0, nan, 1 }, x[2] = { 1, 0 };
        cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
                    A, 2, x, 1);
        r += " trmv=" + num(x[0]);
    }
    {   // dtrsv U N: A = [0], x = [0] (ref: 0)
        double A[1] = { 0 }, x[1] = { 0 };
        cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 1,
                    A, 1, x, 1);
        r += " trsv=" + num(x[0]);
    }
    {   // dsyrk L N: n = 2, k = 1, A = (0, NaN), beta = 1, C21 = 5 (ref: 5)
        double A[2] = { 0, nan }, C[4] = { 1, 5, -1, 1 };
        cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, 2, 1, 1, A, 2,
                    1, C, 2);
        r += " syrk=" + num(C[1]);
    }
    {   // dtrmm L L N N: A = [1 *; NaN 1], B = (0, 1) (ref: B2 = 1)
        double A[4] = { 1, nan, -1, 1 }, B[2] = { 0, 1 };
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, 2, 1, 1, A, 2, B, 2);
        r += " trmm=" + num(B[1]);
    }
    {   // dgemm T N with k = 0, alpha = NaN, beta = 0 (ref: NaN)
        double A[1] = { 1 }, B[1] = { 1 }, C[1] = { 7 };
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, 0, nan, A,
                    1, B, 1, 0, C, 1);
        r += " gemm(k=0)=" + num(C[0]);
    }
    return r;
}

} // namespace

ULMBLAS_TEST_DESIGN(T0602, "conventions/products with zero elements that "
                    "reference BLAS skips", 24,
                    "gemv=0 ger=nan syr=nan spr=nan trmv=nan trsv=nan "
                    "syrk=nan trmm=nan gemm(k=0)=0")
{
    ctx.observed = zeroSkips();
    ctx.pass();
}

//
//  Issue #0023: trmm multiplies B by the structural zeros of the packed
//  diagonal blocks of A, so Inf in B gives NaN in other rows.  Expected:
//  reference BLAS, B = (1, Inf).
//
ULMBLAS_TEST_XFAIL(T0603, "conventions/trmm with Inf in B", 23)
{
    const double inf = std::numeric_limits<double>::infinity();
    double A[4] = { 1, 1, -1, 1 }, B[2] = { 1, inf };   // A = [1 *; 1 1]
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, 2, 1, 1, A, 2, B, 2);
    ctx.check(B[0]==1 && B[1]==inf, "dtrmm: B = (%g, %g), expected (1, inf)",
              B[0], B[1]);
}

//
//  Issue #0025: the C interface accepts the ATLAS extension AtlasConj (114)
//  as transposition argument (reference CBLAS: error 2 or 3); recorded is
//  y of cblas_dgemv and cblas_zgemv with A = [2] or [(2,1)], x = 3.
//
ULMBLAS_TEST_DESIGN(T0604, "conventions/AtlasConj as transposition argument",
                    25, "dgemv=6 zgemv=(6,-3)")
{
    double A[2] = { 2, 1 }, x[2] = { 3, 0 }, y[2] = { 7, 7 };
    double one[2] = { 1, 0 }, zero[2] = { 0, 0 };
    cblas_dgemv(CblasColMajor, CBLAS_TRANSPOSE(114), 1, 1, 1, A, 1, x, 1, 0,
                y, 1);
    std::string r = "dgemv=" + format("%g", y[0]);
    cblas_zgemv(CblasColMajor, CBLAS_TRANSPOSE(114), 1, 1, one, A, 1, x, 1,
                zero, y, 1);
    ctx.observed = r + format(" zgemv=(%g,%g)", y[0], y[1]);
    ctx.pass();
}

//
//  Issue #0026: ?scal with alpha = 0 sets x to zero; reference BLAS
//  multiplies (NaN and Inf give NaN).
//
ULMBLAS_TEST_DESIGN(T0605, "conventions/scal with alpha=0 and NaN, Inf", 26,
                    "0 0")
{
    double x[2] = { std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::infinity() };
    cblas_dscal(2, 0, x, 1);
    ctx.observed = format("%g %g", x[0], x[1]);
    ctx.pass();
}

//
//  Issue #0027: trsm with side = Left multiplies by the reciprocals of the
//  diagonal elements; reference BLAS divides.  1e-300/1e-310: ref 1e+10.
//
ULMBLAS_TEST_DESIGN(T0606, "conventions/trsm divides or multiplies by the "
                    "reciprocal", 27, "inf")
{
    double A[1] = { 1e-310 }, B[1] = { 1e-300 };
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, 1, 1, 1, A, 1, B, 1);
    ctx.observed = format("%g", B[0]);
    ctx.pass();
}

//
//  Issue #0028: the C interface declares complex arguments as float* and
//  double* (CBLAS: void*) and the index type of i?amax as
//  ULMBLAS_INDEX_TYPE (CBLAS: CBLAS_INDEX, size_t in the reference).
//
ULMBLAS_TEST_DESIGN(T0607, "conventions/types of cblas.h", 28,
                    "complex=float* index=signed")
{
    bool floatPtr = std::is_same<decltype(&cblas_cscal),
                                 void (*)(ULMBLAS_INDEX_TYPE, const float *,
                                          float *, ULMBLAS_INDEX_TYPE)>::value;
    bool isSigned = std::is_signed<decltype(cblas_idamax(0, 0, 1))>::value;
    ctx.observed = std::string("complex=") + (floatPtr ? "float*" : "void*")
                 + " index=" + (isSigned ? "signed" : "unsigned");
    ctx.pass();
}

//
//  Issue #0029: the ulmblas_ functions do not check their arguments (no
//  xerbla); recorded: ulmblas_dgemv with ldA = 0 < m = 1 computes.
//
ULMBLAS_TEST_DESIGN(T0608, "conventions/ulmblas_ functions without argument "
                    "checks", 29, "y=6")
{
    double A[1] = { 2 }, x[1] = { 3 }, y[1] = { 7 };
    ulmblas_dgemv(CblasNoTrans, 1, 1, 1, A, 0, x, 1, 0, y, 1);
    ctx.observed = format("y=%g", y[0]);
    ctx.pass();
}
