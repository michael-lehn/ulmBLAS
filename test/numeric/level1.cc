//
//  Numerical tests of the level 1 functions of the C interface.
//
#include <test/numeric/blas.h>
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

namespace {

using namespace test;

const int sizes[]  = { 0, 1, 2, 3, 7, 16, 33, 130 };
const int numSizes = sizeof(sizes)/sizeof(sizes[0]);

// pairs of increments, including negative ones (only for functions with two
// vector arguments, for which BLAS defines negative increments)
struct Incs { int x, y; };

const Incs incs[]  = { {1,1}, {2,3}, {-1,1}, {1,-2}, {-3,-1}, {3,-3} };
const int numIncs  = sizeof(incs)/sizeof(incs[0]);

// positive increments for functions with a single vector argument
const int posIncs[]  = { 1, 2, 5 };
const int numPosIncs = sizeof(posIncs)/sizeof(posIncs[0]);

template <typename T>
bool
same(const Vector<T> &x, const Vector<T> &y)
{
    for (int i=0; i<x.n; ++i) {
        if (!sameBits(x(i), y(i))) {
            return false;
        }
    }
    return true;
}

//
//  AXPY: y = alpha*x + y
//
template <typename T>
void
testAxpy(Context &ctx)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int d=0; d<numSizes; ++d) {
        for (int i=0; i<numIncs; ++i) {
            for (std::size_t a=0; a<s.size(); ++a) {
                int        n = sizes[d];
                Vector<T>  x(n, incs[i].x), y(n, incs[i].y);
                x.randomize(random);
                y.randomize(random);

                Full       one(1, 1);
                one(0,0) = 1;
                Result     ref = multiply(toCdouble(s[a]), fullVector(x), one,
                                          1, fullVector(y));
                Vector<T>  y0 = y;

                blas::axpy(n, s[a], x.ptr(), x.inc, y.ptr(), y.inc);

                std::string what = format("axpy(n=%d,incX=%d,incY=%d,"
                                          "alpha=%s)", n, x.inc, y.inc,
                                          str(s[a]).c_str());
                if (!checkVector(ctx, y, ref, 1, what)
                 || !ctx.check(y.gapsUnchanged(y0), "%s: gaps of y modified",
                               what.c_str()))
                {
                    return;
                }
            }
        }
    }
}

//
//  COPY: y = x,  SWAP: x <-> y  (results must be exact)
//
template <typename T>
void
testCopySwap(Context &ctx, bool swap)
{
    Random random;

    for (int d=0; d<numSizes; ++d) {
        for (int i=0; i<numIncs; ++i) {
            int        n = sizes[d];
            Vector<T>  x(n, incs[i].x), y(n, incs[i].y);
            x.randomize(random);
            y.randomize(random);

            Vector<T>  x0 = x, y0 = y;

            if (swap) {
                blas::swap(n, x.ptr(), x.inc, y.ptr(), y.inc);
            } else {
                blas::copy(n, x.ptr(), x.inc, y.ptr(), y.inc);
            }

            std::string what = format("%s(n=%d,incX=%d,incY=%d), %s",
                                      swap ? "swap" : "copy", n, x.inc, y.inc,
                                      typeName<T>());
            bool ok = swap ? same(x, y0) && same(y, x0)
                           : same(x, x0) && same(y, x0);
            if (!ctx.check(ok, "%s: wrong result", what.c_str())
             || !ctx.check(x.gapsUnchanged(x0) && y.gapsUnchanged(y0),
                           "%s: gaps modified", what.c_str()))
            {
                return;
            }
        }
    }
}

//
//  SCAL: x = alpha*x (alpha of type T, or real for CSSCAL/ZDSCAL)
//
template <typename T, typename Alpha>
bool
scalCase(Context &ctx, Random &random, int n, int inc, Alpha alpha)
{
    Vector<T>  x(n, inc);
    x.randomize(random);

    Vector<T>  x0 = x;
    Full       one(1, 1);
    one(0,0) = 1;
    Result     ref = multiply(toCdouble(alpha), fullVector(x), one, 0, one);

    std::string what = format("%s(n=%d,incX=%d,alpha=%s), %s",
                              std::is_same<T,Alpha>::value ? "scal" : "rscal",
                              n, inc, str(alpha).c_str(), typeName<T>());
    if (std::is_same<T,Alpha>::value) {
        blas::scal(n, T(alpha), x.ptr(), inc);
    } else {
        blas::rscal(n, typename Real<T>::type(realPart(alpha)), x.ptr(), inc);
    }
    return checkVector(ctx, x, ref, 1, what)
        && ctx.check(x.gapsUnchanged(x0), "%s: gaps modified", what.c_str());
}

template <typename T>
void
testScal(Context &ctx)
{
    typedef typename Real<T>::type R;

    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int d=0; d<numSizes; ++d) {
        for (int i=0; i<numPosIncs; ++i) {
            for (std::size_t a=0; a<s.size(); ++a) {
                if (!scalCase<T>(ctx, random, sizes[d], posIncs[i], s[a])) {
                    return;
                }
                if (IsComplex<T>::value
                 && !scalCase<T>(ctx, random, sizes[d], posIncs[i],
                                 R(realPart(s[a]))))
                {
                    return;
                }
            }
        }
    }
}

//
//  DOT, DOTU: x^T y,  DOTC: x^H y
//
template <typename T>
void
testDot(Context &ctx)
{
    Random random;

    for (int c=0; c<(IsComplex<T>::value ? 2 : 1); ++c) {
        for (int d=0; d<numSizes; ++d) {
            for (int i=0; i<numIncs; ++i) {
                int        n = sizes[d];
                Vector<T>  x(n, incs[i].x), y(n, incs[i].y);
                x.randomize(random);
                y.randomize(random);

                Full xt(1, n);
                for (int l=0; l<n; ++l) {
                    xt(0,l) = c ? std::conj(toCdouble(x(l)))
                                : toCdouble(x(l));
                }
                Result  ref = multiply(1, xt, fullVector(y), 0, Full(1,1));
                T       r   = c ? blas::dotc(n, x.ptr(), x.inc, y.ptr(), y.inc)
                                : blas::dotu(n, x.ptr(), x.inc, y.ptr(), y.inc);
                T       e   = fromCdouble<T>(ref.value(0,0));

                if (!ctx.check(closeEnough(r, e, ref.bound(0,0).real(), n),
                               "%s(n=%d,incX=%d,incY=%d) is %s, expected %s, "
                               "%s", c ? "dotc" : "dotu", n, x.inc, y.inc,
                               str(r).c_str(), str(e).c_str(), typeName<T>()))
                {
                    return;
                }
            }
        }
    }
}

//
//  NRM2: sqrt(sum |x_i|^2),  ASUM: sum |re(x_i)| + |im(x_i)|
//
template <typename T>
double
referenceNrm2(const Vector<T> &x)
{
    double scale = 0;
    for (int i=0; i<x.n; ++i) {
        scale = std::max(scale, std::max(std::abs(double(std::real(x(i)))),
                                         std::abs(double(std::imag(x(i))))));
    }
    if (scale==0) {
        return 0;
    }
    double ssq = 0;
    for (int i=0; i<x.n; ++i) {
        double re = double(std::real(x(i)))/scale;
        double im = double(std::imag(x(i)))/scale;
        ssq += re*re + im*im;
    }
    return scale*std::sqrt(ssq);
}

template <typename T>
bool
nrm2Case(Context &ctx, const Vector<T> &x, const char *what)
{
    typedef typename Real<T>::type R;

    R       r = blas::nrm2(x.n, x.ptr(), x.inc);
    double  e = referenceNrm2(x);

    return ctx.check(std::isfinite(r)
                     && std::abs(double(r)-e) <= tolerance<T>(x.n)*e,
                     "nrm2(n=%d,incX=%d) is %g, expected %g (%s), %s",
                     x.n, x.inc, double(r), e, what, typeName<T>());
}

template <typename T>
void
testNrm2(Context &ctx)
{
    Random random;

    for (int d=0; d<numSizes; ++d) {
        for (int i=0; i<numPosIncs; ++i) {
            Vector<T> x(sizes[d], posIncs[i]);
            x.randomize(random);
            if (!nrm2Case(ctx, x, "random")) {
                return;
            }
        }
    }
}

//
//  NRM2 must not overflow or underflow if the result is representable (the
//  reference implementation scales).
//
template <typename T>
void
testNrm2Scaling(Context &ctx)
{
    typedef typename Real<T>::type R;

    Random  random;
    R       big   = std::sqrt(std::numeric_limits<R>::max())*R(4);
    R       small = std::sqrt(std::numeric_limits<R>::min())/R(4);

    for (int d=1; d<numSizes; ++d) {
        Vector<T> x(sizes[d], 1), y(sizes[d], 1);
        x.randomize(random);
        y.randomize(random);
        for (int i=0; i<x.n; ++i) {
            x(i) *= big;
            y(i) *= small;
        }
        if (!nrm2Case(ctx, x, "large entries")
         || !nrm2Case(ctx, y, "tiny entries"))
        {
            return;
        }
    }
}

template <typename T>
void
testAsum(Context &ctx)
{
    Random random;

    for (int d=0; d<numSizes; ++d) {
        for (int i=0; i<numPosIncs; ++i) {
            Vector<T> x(sizes[d], posIncs[i]);
            x.randomize(random);

            double e = 0;
            for (int l=0; l<x.n; ++l) {
                e += std::abs(double(std::real(x(l))))
                   + std::abs(double(std::imag(x(l))));
            }
            double r = blas::asum(x.n, x.ptr(), x.inc);
            if (!ctx.check(std::abs(r-e) <= tolerance<T>(x.n)*e,
                           "asum(n=%d,incX=%d) is %g, expected %g, %s",
                           x.n, x.inc, r, e, typeName<T>()))
            {
                return;
            }
        }
    }
}

//
//  IAMAX: 0-based index of the first element with maximal |re| + |im|
//
template <typename T>
void
testIamax(Context &ctx)
{
    Random random;

    for (int d=1; d<numSizes; ++d) {
        for (int i=0; i<numPosIncs; ++i) {
            for (int ties=0; ties<2; ++ties) {
                int        n = sizes[d];
                Vector<T>  x(n, posIncs[i]);
                x.randomize(random);
                if (ties) {
                    // maximum at two positions, the first one is expected
                    int p = n/3, q = (2*n)/3;
                    x(p) = fromCdouble<T>(cdouble(-2, 1));
                    x(q) = fromCdouble<T>(cdouble(2, IsComplex<T>::value
                                                     ? -1 : 0));
                    if (!IsComplex<T>::value) {
                        x(p) = T(-2);
                    }
                }
                int     e   = 0;
                double  max = -1;
                for (int l=0; l<n; ++l) {
                    double v = std::abs(double(std::real(x(l))))
                             + std::abs(double(std::imag(x(l))));
                    if (v>max) {
                        max = v;
                        e   = l;
                    }
                }
                int r = blas::iamax(n, x.ptr(), x.inc);
                if (!ctx.check(r==e, "iamax(n=%d,incX=%d) is %d, expected "
                               "%d, %s", n, x.inc, r, e, typeName<T>()))
                {
                    return;
                }
            }
        }
    }
}

//
//  ROT: (x,y) = (c*x + s*y, c*y - s*x)
//
template <typename T>
void
testRot(Context &ctx)
{
    Random random;
    T      c = T(0.6), s = T(-0.8);

    for (int d=0; d<numSizes; ++d) {
        for (int i=0; i<numIncs; ++i) {
            int        n = sizes[d];
            Vector<T>  x(n, incs[i].x), y(n, incs[i].y);
            x.randomize(random);
            y.randomize(random);

            Vector<T>  x0 = x, y0 = y;
            blas::rot(n, x.ptr(), x.inc, y.ptr(), y.inc, c, s);

            for (int l=0; l<n; ++l) {
                double ex = double(c)*x0(l) + double(s)*y0(l);
                double ey = double(c)*y0(l) - double(s)*x0(l);
                double bx = std::abs(double(c)*x0(l))+std::abs(double(s)*y0(l));
                double by = std::abs(double(c)*y0(l))+std::abs(double(s)*x0(l));
                if (!ctx.check(closeEnough(x(l), T(ex), bx, 2)
                               && closeEnough(y(l), T(ey), by, 2),
                               "rot(n=%d,incX=%d,incY=%d): element %d wrong, "
                               "%s", n, x.inc, y.inc, l, typeName<T>()))
                {
                    return;
                }
            }
            if (!ctx.check(x.gapsUnchanged(x0) && y.gapsUnchanged(y0),
                           "rot(n=%d,incX=%d,incY=%d): gaps modified", n,
                           x.inc, y.inc))
            {
                return;
            }
        }
    }
}

//
//  ROTG: construct a Givens rotation with c*a + s*b = r, -s*a + c*b = 0.
//  As in reference BLAS, r has the sign of a if |a| > |b| and the sign of b
//  otherwise, and z is s if |a| > |b|, 1/c if c != 0, and 1 otherwise.
//
template <typename T>
void
testRotg(Context &ctx, bool equalAbs)
{
    const T ab[][2]   = { {3,4}, {-4,3}, {4,-3}, {0,2}, {2,0}, {0,0},
                          {-0.5,7}, {1e-3,-2} };
    const T same[][2] = { {1,1}, {-2,2}, {3,-3} };    // |a| == |b|

    int num = equalAbs ? sizeof(same)/sizeof(same[0])
                       : sizeof(ab)/sizeof(ab[0]);

    for (int p=0; p<num; ++p) {
        T a = equalAbs ? same[p][0] : ab[p][0];
        T b = equalAbs ? same[p][1] : ab[p][1];
        T c, s;
        T r = a, z = b;
        blas::rotg(&r, &z, &c, &s);

        double h   = std::sqrt(double(a)*a + double(b)*b);
        double roe = std::abs(a)>std::abs(b) ? a : b;
        double er  = (roe<0 ? -h : h);
        double tol = tolerance<T>(4)*std::max(h, 1.0);
        double ez  = std::abs(a)>std::abs(b) ? double(s)
                   : (c!=0 ? 1/double(c) : 1.0);
        if (h==0) {
            ez = 0;
        }

        bool ok = std::abs(double(r)-er) <= tol
               && std::abs(double(c)*a + double(s)*b - er) <= tol
               && std::abs(-double(s)*a + double(c)*b) <= tol
               && std::abs(double(c)*c + double(s)*s - 1) <= tol
               && std::abs(double(z)-ez) <= tolerance<T>(4)*std::abs(ez);
        if (!ctx.check(ok, "rotg(a=%g,b=%g): r=%g c=%g s=%g z=%g, expected "
                       "r=%g z=%g, %s", double(a), double(b), double(r),
                       double(c), double(s), double(z), er, ez,
                       typeName<T>()))
        {
            return;
        }
    }
}

} // namespace

ULMBLAS_TEST(T0100, "level1/axpy")
{
    testAxpy<float>(ctx); testAxpy<double>(ctx);
    testAxpy<cfloat>(ctx); testAxpy<cdouble>(ctx);
}

ULMBLAS_TEST(T0101, "level1/copy")
{
    testCopySwap<float>(ctx, false); testCopySwap<double>(ctx, false);
    testCopySwap<cfloat>(ctx, false); testCopySwap<cdouble>(ctx, false);
}

ULMBLAS_TEST(T0102, "level1/swap")
{
    testCopySwap<float>(ctx, true); testCopySwap<double>(ctx, true);
    testCopySwap<cfloat>(ctx, true); testCopySwap<cdouble>(ctx, true);
}

ULMBLAS_TEST(T0103, "level1/scal, csscal, zdscal")
{
    testScal<float>(ctx); testScal<double>(ctx);
    testScal<cfloat>(ctx); testScal<cdouble>(ctx);
}

ULMBLAS_TEST(T0104, "level1/dot, dotu, dotc")
{
    testDot<float>(ctx); testDot<double>(ctx);
    testDot<cfloat>(ctx); testDot<cdouble>(ctx);
}

ULMBLAS_TEST(T0105, "level1/nrm2")
{
    testNrm2<float>(ctx); testNrm2<double>(ctx);
    testNrm2<cfloat>(ctx); testNrm2<cdouble>(ctx);
}

ULMBLAS_TEST(T0106, "level1/nrm2 without overflow and underflow")
{
    testNrm2Scaling<float>(ctx); testNrm2Scaling<double>(ctx);
    testNrm2Scaling<cfloat>(ctx); testNrm2Scaling<cdouble>(ctx);
}

ULMBLAS_TEST(T0107, "level1/asum")
{
    testAsum<float>(ctx); testAsum<double>(ctx);
    testAsum<cfloat>(ctx); testAsum<cdouble>(ctx);
}

ULMBLAS_TEST(T0108, "level1/iamax")
{
    testIamax<float>(ctx); testIamax<double>(ctx);
    testIamax<cfloat>(ctx); testIamax<cdouble>(ctx);
}

ULMBLAS_TEST(T0109, "level1/rot")
{
    testRot<float>(ctx); testRot<double>(ctx);
}

ULMBLAS_TEST(T0110, "level1/rotg")
{
    testRotg<float>(ctx, false); testRotg<double>(ctx, false);
}

ULMBLAS_TEST_XFAIL(T0111, "level1/rotg with |a| == |b|", 1)
{
    testRotg<float>(ctx, true); testRotg<double>(ctx, true);
}
