//
//  Tests of conventions that the BLAS documentation leaves open.
//
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
