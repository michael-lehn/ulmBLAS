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

//
//  ROTG with literal expected values (r, c, s, z).  The values were checked
//  against the Netlib reference implementation (refblas/srotg.f,
//  refblas/drotg.f).  Regression test for issue #0001 (z for |a| == |b|).
//
template <typename T>
void
testRotgValues(Context &ctx)
{
    const double q = std::sqrt(2.0), h = 1/std::sqrt(2.0);

    //                    a,  b,  r,     c,    s,    z
    const double t[][6] = { { 1,  1,  q,    h,    h,    q     },  // |a| == |b|
                            {-2,  2,  2*q, -h,    h,   -q     },
                            { 3, -3, -3*q, -h,    h,   -q     },
                            {-1, -1, -q,    h,    h,    q     },
                            { 0,  0,  0,    1,    0,    0     },  // a = b = 0
                            { 4,  3,  5,    0.8,  0.6,  0.6   },  // |a| > |b|
                            {-4,  3, -5,    0.8, -0.6, -0.6   },
                            { 3,  4,  5,    0.6,  0.8,  5./3  },  // |a| < |b|
                            { 3, -4, -5,   -0.6,  0.8, -5./3  },
                            { 0,  2,  2,    0,    1,    1     } };// c == 0

    for (std::size_t k=0; k<sizeof(t)/sizeof(t[0]); ++k) {
        T a = T(t[k][0]), b = T(t[k][1]), c, s;
        T r = a, z = b;
        blas::rotg(&r, &z, &c, &s);

        const T      got[4]  = { r, c, s, z };
        const char  *name[4] = { "r", "c", "s", "z" };
        for (int i=0; i<4; ++i) {
            double e = t[k][2+i];
            if (!ctx.check(std::abs(double(got[i])-e)
                           <= tolerance<T>(2)*std::max(1.0, std::abs(e)),
                           "rotg(a=%g,b=%g): %s=%.9g, expected %.9g, %s",
                           t[k][0], t[k][1], name[i], double(got[i]), e,
                           typeName<T>()))
            {
                return;
            }
        }
    }
}

//
//  ROTM: (x_i, y_i) = H*(x_i, y_i) with H given by param (flag, h11, h21,
//  h12, h22).  The elements of param that the flag does not use are NaN:
//  they must not be referenced.  flag = -2 (H = I) must not change x and y.
//
template <typename T>
void
testRotm(Context &ctx)
{
    const T  nan = test::nan<T>();
    Random   random;

    // hand-computed example: H = [2 3; 4 5]
    {
        T x[2] = { 1, 2 }, y[2] = { 3, 4 };
        const T p[5] = { -1, 2, 4, 3, 5 };
        blas::rotm(2, x, 1, y, 1, p);
        if (!ctx.check(x[0]==11 && x[1]==16 && y[0]==19 && y[1]==28,
                       "rotm(H=[2 3;4 5]): x=(%g,%g) y=(%g,%g), expected "
                       "x=(11,16) y=(19,28), %s", double(x[0]),
                       double(x[1]), double(y[0]), double(y[1]),
                       typeName<T>()))
        {
            return;
        }
    }

    const T flags[] = { -2, -1, 0, 1 };

    for (int f=0; f<4; ++f) {
        for (int d=0; d<numSizes; ++d) {
            for (int i=0; i<numIncs; ++i) {
                int        n = sizes[d];
                Vector<T>  x(n, incs[i].x), y(n, incs[i].y);
                x.randomize(random);
                y.randomize(random);

                T p[5] = { flags[f], nan, nan, nan, nan };
                double h11 = 1, h12 = 0, h21 = 0, h22 = 1;
                if (flags[f]==T(-1)) {
                    for (int k=1; k<5; ++k) {
                        p[k] = random.value<T>();
                    }
                    h11 = p[1]; h21 = p[2]; h12 = p[3]; h22 = p[4];
                } else if (flags[f]==T(0)) {
                    p[2] = random.value<T>();
                    p[3] = random.value<T>();
                    h21 = p[2]; h12 = p[3];
                } else if (flags[f]==T(1)) {
                    p[1] = random.value<T>();
                    p[4] = random.value<T>();
                    h11 = p[1]; h22 = p[4]; h12 = 1; h21 = -1;
                }
                T p0[5];
                std::memcpy(p0, p, sizeof(p));

                Vector<T>  x0 = x, y0 = y;
                blas::rotm(n, x.ptr(), x.inc, y.ptr(), y.inc, p);

                std::string what = format("rotm(n=%d,incX=%d,incY=%d,"
                                          "flag=%g), %s", n, x.inc, y.inc,
                                          double(flags[f]), typeName<T>());
                for (int l=0; l<n; ++l) {
                    double ex = h11*x0(l) + h12*y0(l);
                    double ey = h21*x0(l) + h22*y0(l);
                    double bx = std::abs(h11*x0(l)) + std::abs(h12*y0(l));
                    double by = std::abs(h21*x0(l)) + std::abs(h22*y0(l));
                    bool   ok = flags[f]==T(-2)
                              ? sameBits(x(l), x0(l)) && sameBits(y(l), y0(l))
                              : closeEnough(x(l), T(ex), bx, 2)
                                && closeEnough(y(l), T(ey), by, 2);
                    if (!ctx.check(ok, "%s: element %d wrong",
                                   what.c_str(), l))
                    {
                        return;
                    }
                }
                if (!ctx.check(x.gapsUnchanged(x0) && y.gapsUnchanged(y0),
                               "%s: gaps modified", what.c_str())
                 || !ctx.check(std::memcmp(p, p0, sizeof(p))==0,
                               "%s: param modified", what.c_str()))
                {
                    return;
                }
            }
        }
    }
}

//
//  ROTMG with literal expected values (d1, d2, x1, flag, h11, h21, h12,
//  h22; elements of H that the flag does not use are not set and must not
//  be modified).  The values were computed with DROTMG of Reference-LAPACK
//  (master, commit 6a1a8b8a, 2026-10-25).  They agree with Reference BLAS
//  3.12.1 except for the last case: when both scale factors are rescaled,
//  3.12.1 resets h12 to 1, which master corrects (H*(x1,y1)^T = (x1',0)^T
//  holds only for the corrected value).  The cases cover d1 < 0, d2*y1 = 0,
//  u <= 0, q2 < 0, flag 0 and 1, and rescaling of d1 (up and down) and d2.
//
template <typename T>
void
testRotmgValues(Context &ctx)
{
    const double U = 9;     // marks elements of param that are not set

    //                       input d1, d2, x1, y1;   d1', d2', x1';
    //                       flag, h11, h21, h12, h22
    const double t[][12] = {
        { 4, 1, 1, 1,               3.2, 0.8, 1.25,
          0, U, -1, 0.25, U },
        { 1, 4, 1, 1,               3.2, 0.8, 1.25,
          1, 0.25, U, U, 1 },
        { -1, 1, 1, 1,              0, 0, 0,
          -1, 0, 0, 0, 0 },
        { 1, 1, 1, 0,               1, 1, 1,
          -2, U, U, U, U },
        { 1, -2, 1, 0.9,            0, 0, 0,
          -1, 0, 0, 0, 0 },
        { 1, -1, 0.5, 1,            0, 0, 0,
          -1, 0, 0, 0, 0 },
        { 1e8, 1, 1, 1,             5.9604644179344186, 0.99999999000000017,
                                    4096.0000409599998,
          -1, 4096, -1, 4.0960000000000001e-05, 1 },
        { 1e-9, 1, 1, 1e-6,         0.016760455544455547, 0.99900099900099915,
                                    0.00024438476562499997,
          -1, 0.000244140625, -9.9999999999999995e-07, 0.24414062499999997,
          1 },
        { 1, 1e9, 1, 1e-5,          0.90909090909090906, 54.186040704900563,
                                    1.1000000000000001,
          -1, 1, -0.040960000000000003, 10000, 4096 },
        { 2, 3, -1.5, 0.25,         1.9199999999999999, 2.8799999999999999,
                                    -1.5625,
          0, U, 0.16666666666666666, -0.25, U },
        { 0, 1, 1, 1,               1, 0, 1,
          1, 0, U, U, 1 },
        { 0.5, -3, -2, 0.75,        3.2000000000000002, -19.199999999999999,
                                    -0.3125,
          0, U, 0.375, 2.25, U },
        { 3, 2, 0, 1,               2, 3, 1,
          1, 0, U, U, 0 },
        { 1e-12, 1e-12, 1000, 1000, 8.3886079999999998e-06,
                                    8.3886079999999998e-06, 0.48828125,
          -1, 0.000244140625, -0.000244140625, 0.000244140625,
          0.000244140625 } };

    for (std::size_t k=0; k<sizeof(t)/sizeof(t[0]); ++k) {
        T d1 = T(t[k][0]), d2 = T(t[k][1]), x1 = T(t[k][2]);
        T p[5] = { T(U), T(U), T(U), T(U), T(U) };
        blas::rotmg(&d1, &d2, &x1, T(t[k][3]), p);

        const T      got[8]  = { d1, d2, x1, p[0], p[1], p[2], p[3], p[4] };
        const char  *name[8] = { "d1", "d2", "x1", "flag", "h11", "h21",
                                 "h12", "h22" };
        for (int i=0; i<8; ++i) {
            double e = t[k][4+i];
            if (!ctx.check(std::abs(double(got[i])-e)
                           <= tolerance<T>(4)*std::abs(e),
                           "rotmg(d1=%g,d2=%g,x1=%g,y1=%g): %s=%.9g, "
                           "expected %.9g, %s", t[k][0], t[k][1], t[k][2],
                           t[k][3], name[i], double(got[i]), e,
                           typeName<T>()))
            {
                return;
            }
        }
    }
}

//
//  ROTMG with random arguments over a wide range of magnitudes (with
//  rescaling): for the computed H (see testRotm), d1', d2', x1'
//
//      H*(x1, y1)^T = (x1', 0)^T   and   H^T*diag(d1',d2')*H = diag(d1,d2),
//
//  i.e. diag(d1',d2')^(1/2)*H*diag(d1,d2)^(-1/2) is orthogonal.
//
template <typename T>
void
testRotmgProperties(Context &ctx)
{
    Random random;

    for (int k=0; k<2000; ++k) {
        // magnitudes 2^-40 ... 2^40 (also for float, so that d1 and d2 are
        // rescaled in both directions); d2 of both signs
        double e1 = std::floor(40*random.uniform());
        double e2 = std::floor(40*random.uniform());
        double e3 = std::floor(10*random.uniform());
        T d1 = T(std::ldexp(std::abs(random.uniform())+0.01, int(e1)));
        T d2 = T(std::ldexp(random.uniform(), int(e2)));
        T x1 = T(std::ldexp(random.uniform(), int(e3)));
        T y1 = T(random.uniform());

        const T D1 = d1, D2 = d2, X1 = x1;
        T p[5] = { 0, 0, 0, 0, 0 };
        blas::rotmg(&d1, &d2, &x1, y1, p);

        std::string what = format("rotmg(d1=%.9g,d2=%.9g,x1=%.9g,y1=%.9g), "
                                  "%s", double(D1), double(D2), double(X1),
                                  double(y1), typeName<T>());
        double h11, h12, h21, h22;
        if (p[0]==T(-2)) {
            if (!ctx.check(D2*y1==T(0) && sameBits(d1, D1)
                           && sameBits(d2, D2) && sameBits(x1, X1),
                           "%s: flag -2 for d2*y1 != 0 or arguments "
                           "modified", what.c_str()))
            {
                return;
            }
            continue;
        } else if (p[0]==T(-1)) {
            h11 = p[1]; h21 = p[2]; h12 = p[3]; h22 = p[4];
        } else if (p[0]==T(0)) {
            h11 = 1; h21 = p[2]; h12 = p[3]; h22 = 1;
        } else if (p[0]==T(1)) {
            h11 = p[1]; h21 = -1; h12 = 1; h22 = p[4];
        } else {
            ctx.check(false, "%s: flag %g", what.c_str(), double(p[0]));
            return;
        }
        if (p[0]==T(-1) && h11==0 && h12==0 && h21==0 && h22==0) {
            // no transformation exists (u <= 0 or q2 < 0, d2 < 0 only)
            if (!ctx.check(D2<T(0) && d1==T(0) && d2==T(0) && x1==T(0),
                           "%s: H zeroed for d2 >= 0", what.c_str())) {
                return;
            }
            continue;
        }
        const double a = D1, b = D2, x = X1, y = y1;
        const double c1 = d1, c2 = d2, z = x1;
        // H*(x1,y1)^T = (x1',0)^T
        bool ok = closeEnough(T(h11*x + h12*y), T(z),
                              std::abs(h11*x) + std::abs(h12*y), 4)
               && closeEnough(T(h21*x + h22*y), T(0),
                              std::abs(h21*x) + std::abs(h22*y), 4)
        // H^T*diag(d1',d2')*H = diag(d1,d2)
               && closeEnough(T(c1*h11*h11 + c2*h21*h21), T(a),
                              std::abs(c1*h11*h11) + std::abs(c2*h21*h21), 4)
               && closeEnough(T(c1*h11*h12 + c2*h21*h22), T(0),
                              std::abs(c1*h11*h12) + std::abs(c2*h21*h22), 4)
               && closeEnough(T(c1*h12*h12 + c2*h22*h22), T(b),
                              std::abs(c1*h12*h12) + std::abs(c2*h22*h22), 4);
        if (!ctx.check(ok, "%s: flag=%g d1'=%.9g d2'=%.9g x1'=%.9g H=[%.9g "
                       "%.9g; %.9g %.9g] does not satisfy H*(x1,y1)=(x1',0) "
                       "and H^T*D'*H=D", what.c_str(), double(p[0]), c1, c2,
                       z, h11, h12, h21, h22))
        {
            return;
        }
    }
}

//
//  SDSDOT: alpha + x^T y,  DSDOT: x^T y for single precision x, y, both
//  accumulated in double precision.  The products of single precision
//  numbers are exact in double precision, so the result is compared with a
//  double precision sum using the error bound of double precision (a sum
//  in single precision fails this).  SDSDOT rounds the result to single
//  precision once.
//
void
testSdsdot(Context &ctx)
{
    // 2^24 + 1 - 2^24 is 0 in single and 1 in double precision
    {
        const float x[3] = { 16777216.0f, 1.0f, -16777216.0f };
        const float y[3] = { 1.0f, 1.0f, 1.0f };
        double d = cblas_dsdot(3, x, 1, y, 1);
        float  s = cblas_sdsdot(3, 0.5f, x, 1, y, 1);
        double r = cblas_dsdot(3, x, -1, y, 1);
        if (!ctx.check(d==1.0 && s==1.5f && r==1.0,
                       "dsdot(2^24, 1, -2^24) = %.17g (incX=-1: %.17g), "
                       "sdsdot(0.5; ...) = %.9g, expected 1 and 1.5", d, r,
                       double(s)))
        {
            return;
        }
    }
    // n <= 0: dsdot = 0, sdsdot = alpha
    {
        const float x[1] = { 2 }, y[1] = { 3 };
        if (!ctx.check(cblas_dsdot(0, x, 1, y, 1)==0.0
                       && cblas_dsdot(-1, x, 1, y, 1)==0.0
                       && cblas_sdsdot(0, 0.25f, x, 1, y, 1)==0.25f
                       && cblas_sdsdot(-1, 0.25f, x, 1, y, 1)==0.25f,
                       "dsdot or sdsdot with n <= 0"))
        {
            return;
        }
    }

    Random      random;
    const float alphas[] = { 0.0f, 1.0f, -0.625f, 3.0e5f };

    for (int d=0; d<numSizes; ++d) {
        for (int i=0; i<numIncs; ++i) {
            for (int a=0; a<4; ++a) {
                int            n = sizes[d];
                Vector<float>  x(n, incs[i].x), y(n, incs[i].y);
                x.randomize(random);
                y.randomize(random);
                // large and small entries
                for (int l=0; l<n; l+=3) {
                    x(l) *= 4096.0f;
                }
                Vector<float>  x0 = x, y0 = y;

                double e = 0, bound = 0;
                for (int l=0; l<n; ++l) {
                    e     += double(x(l))*double(y(l));
                    bound += std::abs(double(x(l))*double(y(l)));
                }
                double tol = tolerance<double>(n)*bound
                           + std::numeric_limits<double>::min();

                double dr = cblas_dsdot(n, x.ptr(), x.inc, y.ptr(), y.inc);
                float  sr = cblas_sdsdot(n, alphas[a], x.ptr(), x.inc,
                                         y.ptr(), y.inc);
                double es = double(alphas[a]) + e;
                double ts = tol + std::abs(double(alphas[a]))*0
                          + double(eps<float>())*std::abs(es)
                          + std::numeric_limits<float>::min();

                if (!ctx.check(std::abs(dr-e)<=tol,
                               "dsdot(n=%d,incX=%d,incY=%d) = %.17g, "
                               "expected %.17g (bound %g)", n, x.inc, y.inc,
                               dr, e, tol)
                 || !ctx.check(std::abs(double(sr)-es)<=ts,
                               "sdsdot(n=%d,alpha=%g,incX=%d,incY=%d) = "
                               "%.9g, expected %.17g", n, double(alphas[a]),
                               x.inc, y.inc, double(sr), es)
                 || !ctx.check(std::memcmp(&x.data[0], &x0.data[0],
                                           x.data.size()*sizeof(float))==0
                               && std::memcmp(&y.data[0], &y0.data[0],
                                              y.data.size()*sizeof(float))==0,
                               "sdsdot or dsdot(n=%d,incX=%d,incY=%d) "
                               "modified x or y", n, x.inc, y.inc))
                {
                    return;
                }
            }
        }
    }
}

//
//  Complex ROTG (CROTG, ZROTG): c real, c*a + s*b = r, -conj(s)*a + c*b = 0,
//  c^2 + |s|^2 = 1, r with the phase of a (r = |b| and s = conj(b)/|b| if
//  a = 0, c = 1 and s = 0 if b = 0), as in the Reference BLAS since LAPACK
//  3.10.  b is not modified.
//
template <typename T>
void
testComplexRotg(Context &ctx)
{
    typedef typename Real<T>::type R;

    const double h = 1/std::sqrt(2.0), q = std::sqrt(2.0);

    //                          a,       b,         r,       c,   s
    const double t[][9] = { { 3, 4,    0, 0,      3, 4,    1,   0, 0    },
                            { 0, 0,    3, -4,     5, 0,    0,   0.6, 0.8 },
                            { 3, 0,    0, 4,      5, 0,    0.6, 0, -0.8 },
                            { 1, 1,    1, -1,     q, q,    h,   0, h    },
                            { -2, 0,   0, -2,    -2*q, 0,  h,   0, -h   },
                            { 0, 0,    0, 0,      0, 0,    1,   0, 0    } };

    for (std::size_t k=0; k<sizeof(t)/sizeof(t[0]); ++k) {
        T a = T(R(t[k][0]), R(t[k][1]));
        T b = T(R(t[k][2]), R(t[k][3]));
        T s;
        R c;
        T r = a, b1 = b;
        blas::rotg(&r, &b1, &c, &s);

        double tol = tolerance<T>(4)*4;
        bool ok = std::abs(toCdouble(r) - cdouble(t[k][4], t[k][5])) <= tol*4
               && std::abs(double(c) - t[k][6]) <= tol
               && std::abs(toCdouble(s) - cdouble(t[k][7], t[k][8])) <= tol
               && sameBits(b1, b);
        if (!ctx.check(ok, "rotg(a=%s,b=%s): r=%s c=%g s=%s, expected "
                       "r=(%g,%g) c=%g s=(%g,%g), %s", str(a).c_str(),
                       str(b).c_str(), str(r).c_str(), double(c),
                       str(s).c_str(), t[k][4], t[k][5], t[k][6], t[k][7],
                       t[k][8], typeName<T>()))
        {
            return;
        }
    }

    // random a and b with magnitudes up to 2^(+-100) (2^(+-60) for float):
    // |a|^2 + |b|^2 must not overflow or underflow
    Random    random;
    const int maxExp = std::is_same<R,float>::value ? 60 : 100;

    for (int k=0; k<1000; ++k) {
        int ea = int(std::floor(maxExp*random.uniform()));
        int eb = int(std::floor(maxExp*random.uniform()));
        T a = random.value<T>()*R(std::ldexp(1.0, ea));
        T b = random.value<T>()*R(std::ldexp(1.0, eb));
        if (k%10==0) {
            b = T(0);
        } else if (k%10==1) {
            a = T(0);
        }
        T r = a, b1 = b, s;
        R c;
        blas::rotg(&r, &b1, &c, &s);

        cdouble A = toCdouble(a), B = toCdouble(b), S = toCdouble(s),
                Rr = toCdouble(r);
        double  C  = c, scale = std::abs(A) + std::abs(B);
        double  tol = tolerance<T>(4);
        if (scale==0) {
            scale = 1;
        }
        // phase of r equal to the phase of a: r*conj(a) real and >= 0
        cdouble phase = Rr*std::conj(A);
        bool ok = std::abs(C*A + S*B - Rr) <= tol*scale
               && std::abs(-std::conj(S)*A + C*B) <= tol*scale
               && std::abs(C*C + std::norm(S) - 1) <= tol
               && std::abs(std::abs(Rr) - std::hypot(std::abs(A),
                                                     std::abs(B)))
                  <= tol*scale
               && C>=0 && C<=1
               && std::abs(phase.imag()) <= tol*std::abs(phase)
               && phase.real() >= 0
               && isFinite(r) && isFinite(s) && std::isfinite(C)
               && sameBits(b1, b);
        if (!ctx.check(ok, "rotg(a=%s,b=%s): r=%s c=%g s=%s, %s",
                       str(a).c_str(), str(b).c_str(), str(r).c_str(),
                       double(c), str(s).c_str(), typeName<T>()))
        {
            return;
        }
    }
}

//
//  CSROT, ZDROT: (x,y) = (c*x + s*y, c*y - s*x) with real c, s
//
template <typename T>
void
testComplexRot(Context &ctx)
{
    typedef typename Real<T>::type R;

    Random random;
    const R cs[][2] = { { R(0.6), R(-0.8) }, { 1, 0 }, { 0, 1 },
                        { R(-0.28), R(0.96) } };

    for (int k=0; k<4; ++k) {
        R c = cs[k][0], s = cs[k][1];
        for (int d=0; d<numSizes; ++d) {
            for (int i=0; i<numIncs; ++i) {
                int        n = sizes[d];
                Vector<T>  x(n, incs[i].x), y(n, incs[i].y);
                x.randomize(random);
                y.randomize(random);

                Vector<T>  x0 = x, y0 = y;
                blas::rot(n, x.ptr(), x.inc, y.ptr(), y.inc, c, s);

                for (int l=0; l<n; ++l) {
                    cdouble X = toCdouble(x0(l)), Y = toCdouble(y0(l));
                    cdouble ex = double(c)*X + double(s)*Y;
                    cdouble ey = double(c)*Y - double(s)*X;
                    double  bx = std::abs(double(c)*X)+std::abs(double(s)*Y);
                    double  by = std::abs(double(c)*Y)+std::abs(double(s)*X);
                    if (!ctx.check(closeEnough(x(l), fromCdouble<T>(ex), bx, 2)
                                   && closeEnough(y(l), fromCdouble<T>(ey),
                                                  by, 2),
                                   "rot(n=%d,incX=%d,incY=%d,c=%g,s=%g): "
                                   "element %d wrong, %s", n, x.inc, y.inc,
                                   double(c), double(s), l, typeName<T>()))
                    {
                        return;
                    }
                }
                if (!ctx.check(x.gapsUnchanged(x0) && y.gapsUnchanged(y0),
                               "rot(n=%d,incX=%d,incY=%d): gaps modified, "
                               "%s", n, x.inc, y.inc, typeName<T>()))
                {
                    return;
                }
            }
        }
    }
}

//
//  SCABS1, DCABS1: |re(z)| + |im(z)| (exact)
//
template <typename T>
void
testCabs1(Context &ctx)
{
    typedef typename Real<T>::type R;

    const double t[][3] = { { 3, -4, 7 }, { -1.5, 2.25, 3.75 }, { 0, 0, 0 },
                            { -0.0, -0.0, 0 }, { -5, 0, 5 }, { 0, -6, 6 },
                            { 1e30, -1e30, 2e30 } };

    for (std::size_t k=0; k<sizeof(t)/sizeof(t[0]); ++k) {
        T z = T(R(t[k][0]), R(t[k][1]));
        R r = blas::abs1(z);
        R e = std::abs(R(t[k][0])) + std::abs(R(t[k][1]));
        if (!ctx.check(r==e, "abs1(%s) = %.9g, expected %.9g, %s",
                       str(z).c_str(), double(r), double(e), typeName<T>()))
        {
            return;
        }
    }
}

//
//  NRM2 with Inf and NaN: NaN if an element (or a real or imaginary part)
//  is NaN, otherwise Inf if one is infinite, as in the Reference BLAS since
//  LAPACK 3.10.  Two infinite elements must not give Inf/Inf = NaN.
//
template <typename T>
void
testNrm2InfNaN(Context &ctx)
{
    typedef typename Real<T>::type R;

    const R  inf = std::numeric_limits<R>::infinity();
    const R  nan = std::numeric_limits<R>::quiet_NaN();
    const bool cplx = IsComplex<T>::value;

    // patterns: value of element 0 and of element p, the others are 1;
    // expected 'I' (Inf) or 'N' (NaN)
    struct Pattern { double re0, im0, rep, imp; char expected; };
    const double I = std::numeric_limits<double>::infinity();
    const double N = std::numeric_limits<double>::quiet_NaN();
    const Pattern pattern[] = { { I, 0, 1, 0, 'I' },  { 1, 0, I, 0, 'I' },
                                { I, 0, I, 0, 'I' },  { -I, 0, I, 0, 'I' },
                                { I, 0, -I, 0, 'I' }, { N, 0, 1, 0, 'N' },
                                { 1, 0, N, 0, 'N' },  { I, 0, N, 0, 'N' },
                                { N, 0, I, 0, 'N' },  { 0, 0, I, 0, 'I' },
                                { 0, I, 1, 0, 'I' },  { I, I, 1, 0, 'I' },
                                { I, N, 1, 0, 'N' },  { N, I, 1, 0, 'N' },
                                { 1, I, 1, -I, 'I' }, { 1, 0, 0, N, 'N' } };

    for (std::size_t k=0; k<sizeof(pattern)/sizeof(pattern[0]); ++k) {
        const Pattern &q = pattern[k];
        if (!cplx && (q.im0!=0 || q.imp!=0)) {
            continue;
        }
        for (int n=1; n<=5; ++n) {
            for (int p=(n==1 ? 0 : 1); p<n; ++p) {
                if (n==1 && (q.rep!=1 || q.imp!=0)) {
                    continue;       // element p would overwrite element 0
                }
                for (int i=0; i<numPosIncs; ++i) {
                    Vector<T> x(n, posIncs[i]);
                    for (int l=0; l<n; ++l) {
                        x(l) = fromCdouble<T>(cdouble(1, cplx ? 1 : 0));
                    }
                    x(0) = fromCdouble<T>(cdouble(q.re0, q.im0));
                    if (n>1) {
                        x(p) = fromCdouble<T>(cdouble(q.rep, q.imp));
                    }
                    R r = blas::nrm2(n, x.ptr(), x.inc);
                    bool ok = q.expected=='I' ? r==inf : std::isnan(r);
                    (void)nan;
                    if (!ctx.check(ok, "nrm2(n=%d,incX=%d) with x(0)=%s, "
                                   "x(%d)=%s is %g, expected %s, %s", n,
                                   x.inc, str(x(0)).c_str(), p,
                                   str(x(p)).c_str(), double(r),
                                   q.expected=='I' ? "Inf" : "NaN",
                                   typeName<T>()))
                    {
                        return;
                    }
                }
            }
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

ULMBLAS_TEST(T0111, "level1/rotg with |a| == |b|")
{
    testRotg<float>(ctx, true); testRotg<double>(ctx, true);
}

ULMBLAS_TEST(T0112, "level1/rotg reference values")
{
    testRotgValues<float>(ctx); testRotgValues<double>(ctx);
}

ULMBLAS_TEST(T0113, "level1/srotm, drotm")
{
    testRotm<float>(ctx); testRotm<double>(ctx);
}

ULMBLAS_TEST(T0114, "level1/srotmg, drotmg reference values")
{
    testRotmgValues<float>(ctx); testRotmgValues<double>(ctx);
}

ULMBLAS_TEST(T0115, "level1/srotmg, drotmg: H*(x1,y1) = (x1',0) and "
             "H^T*D'*H = D, rescaling")
{
    testRotmgProperties<float>(ctx); testRotmgProperties<double>(ctx);
}

ULMBLAS_TEST(T0116, "level1/sdsdot, dsdot (double precision accumulation)")
{
    testSdsdot(ctx);
}

ULMBLAS_TEST(T0117, "level1/crotg, zrotg")
{
    testComplexRotg<cfloat>(ctx); testComplexRotg<cdouble>(ctx);
}

ULMBLAS_TEST(T0118, "level1/csrot, zdrot")
{
    testComplexRot<cfloat>(ctx); testComplexRot<cdouble>(ctx);
}

ULMBLAS_TEST(T0119, "level1/scabs1, dcabs1")
{
    testCabs1<cfloat>(ctx); testCabs1<cdouble>(ctx);
}

ULMBLAS_TEST(T0120, "level1/nrm2 with Inf and NaN")
{
    testNrm2InfNaN<float>(ctx); testNrm2InfNaN<double>(ctx);
    testNrm2InfNaN<cfloat>(ctx); testNrm2InfNaN<cdouble>(ctx);
}
