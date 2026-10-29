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
//  the operands multiplied by alpha (A, B, x, y).  Reference BLAS returns
//  immediately in this case and does not reference these operands.  Returns
//  the names of the functions that propagate the NaN into the result.
//
template <typename T>
std::string
nanPropagation(const char *prefix)
{
    const int    n   = 4;
    const T      nan = test::nan<T>();
    const T      one(1);
    std::vector<T>  bad(n*n, nan);
    std::vector<T>  out;
    std::string     result;
    bool            herm = IsComplex<T>::value;
    CBLAS_ORDER     o = CblasColMajor;
    CBLAS_UPLO      u = CblasLower;
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
//  0-based indices returned by i?amax for vectors with NaN and Inf:
//  (1, NaN, 5, -7, 2), (NaN, 2, 3) and (2, Inf, NaN, 9).  For complex types
//  the real parts are given, the imaginary parts are zero.
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
    const R *v[3] = { v1, v2, v3 };
    const int n[3] = { 5, 3, 4 };

    std::string result = std::string(prefix) + ":";
    for (int k=0; k<3; ++k) {
        std::vector<T> x(n[k]);
        for (int i=0; i<n[k]; ++i) {
            x[i] = T(v[k][i]);
        }
        result += format(" %d", blas::iamax(n[k], &x[0], 1));
    }
    return result;
}

} // namespace

ULMBLAS_TEST_DESIGN(T0600, "conventions/alpha=0 with NaN in unreferenced "
                    "operands", 2,
                    "sger ssymv ssyr ssyr2 sspr sspr2; "
                    "dger dsymv dsyr dsyr2 dspr dspr2; "
                    "cgeru cgerc chemv; zgeru zgerc zhemv")
{
    ctx.observed = nanPropagation<float>("s") + "; "
                 + nanPropagation<double>("d") + "; "
                 + nanPropagation<cfloat>("c") + "; "
                 + nanPropagation<cdouble>("z");
    ctx.pass();
}

ULMBLAS_TEST_DESIGN(T0601, "conventions/i?amax with NaN and Inf", 14,
                    "s: 3 0 1; d: 3 0 1; c: 3 0 1; z: 3 0 1")
{
    ctx.observed = iamaxWithNan<float>("s") + "; "
                 + iamaxWithNan<double>("d") + "; "
                 + iamaxWithNan<cfloat>("c") + "; "
                 + iamaxWithNan<cdouble>("z");
    ctx.pass();
}
