//
//  Conformance of the C interface with the reference CBLAS of LAPACK 3.12.1
//  (BLAS/CBLAS conformance audit, see test/known_issues.md #0017, #0018,
//  #0021, #0022):
//
//  - level 1 functions with increments <= 0: cblas_?asum and cblas_i?amax
//    return 0, cblas_?scal does nothing, cblas_?nrm2 supports negative
//    increments; the ulmblas_ functions use the general strides of ulmBLAS
//    (BLAS storage order for negative increments, x[0] repeated for zero);
//  - cblas_?her2k with row major storage does not write to alpha;
//  - ?rotg as in LAPACK 3.12.1 (no overflow for large a, b; Inf; signs).
//
//  The vectors lie in the middle of buffers whose other elements are guard
//  values; the complete buffers are compared bitwise after each call, and
//  results that read a guard value would be wrong.
//
#include <csetjmp>
#include <csignal>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include <test/numeric/blas.h>
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

namespace {

using namespace test;

//
//  ulmblas_ level 1 functions for all four types
//
namespace ulm {

inline float  asum(int n, const float *x, int ix)   { return ulmblas_sasum(n, x, ix); }
inline double asum(int n, const double *x, int ix)  { return ulmblas_dasum(n, x, ix); }
inline float  asum(int n, const cfloat *x, int ix)  { return ulmblas_scasum(n, blas::rp(x), ix); }
inline double asum(int n, const cdouble *x, int ix) { return ulmblas_dzasum(n, blas::rp(x), ix); }

inline float  nrm2(int n, const float *x, int ix)   { return ulmblas_snrm2(n, x, ix); }
inline double nrm2(int n, const double *x, int ix)  { return ulmblas_dnrm2(n, x, ix); }
inline float  nrm2(int n, const cfloat *x, int ix)  { return ulmblas_scnrm2(n, blas::rp(x), ix); }
inline double nrm2(int n, const cdouble *x, int ix) { return ulmblas_dznrm2(n, blas::rp(x), ix); }

inline int iamax(int n, const float *x, int ix)   { return ulmblas_isamax(n, x, ix); }
inline int iamax(int n, const double *x, int ix)  { return ulmblas_idamax(n, x, ix); }
inline int iamax(int n, const cfloat *x, int ix)  { return ulmblas_icamax(n, blas::rp(x), ix); }
inline int iamax(int n, const cdouble *x, int ix) { return ulmblas_izamax(n, blas::rp(x), ix); }

inline void scal(int n, float a, float *x, int ix)   { ulmblas_sscal(n, a, x, ix); }
inline void scal(int n, double a, double *x, int ix) { ulmblas_dscal(n, a, x, ix); }
inline void scal(int n, cfloat a, cfloat *x, int ix)
{ ulmblas_cscal(n, blas::rp(&a), blas::rp(x), ix); }
inline void scal(int n, cdouble a, cdouble *x, int ix)
{ ulmblas_zscal(n, blas::rp(&a), blas::rp(x), ix); }

} // namespace ulm

//
//  Buffer of 32 elements with guard values; the vector starts at element
//  16 (the start of its storage: for a negative increment its last element
//  in BLAS order is at element 16).
//
template <typename T>
struct GuardedVector
{
    explicit
    GuardedVector(int seed)
        : buf(32)
    {
        for (int i=0; i<32; ++i) {
            buf[i] = fromCdouble<T>(cdouble(1000+i+seed, -(i%5)-seed));
        }
    }

    T *
    x()
    {
        return &buf[16];
    }

    // element e (0-based, BLAS order) of a vector of length n
    T &
    elem(int e, int n, int inc)
    {
        return x()[inc>=0 ? e*inc : (n-1-e)*(-inc)];
    }

    bool
    sameAs(const GuardedVector &other) const
    {
        for (int i=0; i<32; ++i) {
            if (!sameBits(buf[i], other.buf[i])) {
                return false;
            }
        }
        return true;
    }

    std::vector<T> buf;
};

template <typename T>
double
abs1d(const T &x)
{
    return std::abs(double(std::real(x))) + std::abs(double(std::imag(x)));
}

//
//  The values 1, -5, 3 (complex: imaginary parts 0, 2, -1) at the logical
//  positions of a vector of length 3 with increment inc.
//
template <typename T>
GuardedVector<T>
sample(int inc)
{
    GuardedVector<T> v(0);
    const cdouble val[3] = { cdouble(1, 0), cdouble(-5, 2), cdouble(3, -1) };
    int n = inc==0 ? 1 : 3;
    for (int e=0; e<n; ++e) {
        v.elem(e, 3, inc) = fromCdouble<T>(val[e]);
    }
    return v;
}

template <typename T>
void
testLevel1Cblas(Context &ctx)
{
    const char *t = typeName<T>();
    const int   incs[] = { 0, -1, -2 };
    const int   ns[]   = { 1, 3 };

    for (int i=0; i<3; ++i) {
        for (int j=0; j<2; ++j) {
            int inc = incs[i], n = ns[j];

            GuardedVector<T> x = sample<T>(inc), x0 = x;
            ctx.check(blas::asum(n, x.x(), inc)==0 && x.sameAs(x0),
                      "cblas asum(n=%d, incX=%d) != 0, %s", n, inc, t);
            ctx.check(blas::iamax(n, x.x(), inc)==0 && x.sameAs(x0),
                      "cblas i?amax(n=%d, incX=%d) != 0, %s", n, inc, t);
            blas::scal(n, fromCdouble<T>(cdouble(2, 1)), x.x(), inc);
            ctx.check(x.sameAs(x0), "cblas scal(n=%d, incX=%d) modified "
                      "the buffer, %s", n, inc, t);
            blas::rscal(n, typename Real<T>::type(2), x.x(), inc);
            ctx.check(x.sameAs(x0), "cblas real scal(n=%d, incX=%d) "
                      "modified the buffer, %s", n, inc, t);
        }
    }
    // cblas_i?amax with n <= 0: 0 (reference CBLAS), for all increments
    for (int n=-1; n<=0; ++n) {
        GuardedVector<T> x = sample<T>(1);
        ctx.check(blas::iamax(n, x.x(), 1)==0,
                  "cblas i?amax(n=%d) != 0, %s", n, t);
    }
    // positive increments are unaffected
    {
        GuardedVector<T> x = sample<T>(2), x0 = x;
        double s = abs1d(x.elem(0,3,2)) + abs1d(x.elem(1,3,2))
                 + abs1d(x.elem(2,3,2));
        ctx.check(std::abs(blas::asum(3, x.x(), 2)-s)<=1e-5*s
                  && blas::iamax(3, x.x(), 2)==1 && x.sameAs(x0),
                  "cblas asum/i?amax with incX=2, %s", t);
    }
    // cblas_?nrm2: negative increments as in LAPACK 3.12.1, zero increment
    // gives sqrt(n)*|x_0|
    for (int i=0; i<3; ++i) {
        int inc = incs[i], n = 3;
        GuardedVector<T> x = sample<T>(inc), x0 = x;
        double s = 0;
        for (int e=0; e<n; ++e) {
            double a = std::abs(toCdouble(x.elem(inc==0 ? 0 : e, n, inc)));
            s += a*a;
        }
        double r = blas::nrm2(n, x.x(), inc);
        ctx.check(std::abs(r-std::sqrt(s))<=1e-5*std::sqrt(s)
                  && x.sameAs(x0),
                  "cblas nrm2(n=%d, incX=%d) is %g, expected %g, %s", n, inc,
                  r, std::sqrt(s), t);
    }
}

//
//  ulmblas_ functions: general strides (a negative increment traverses the
//  storage backwards from its end, a zero increment repeats x[0])
//
template <typename T>
void
testLevel1Ulmblas(Context &ctx)
{
    const char *t = typeName<T>();

    for (int inc : { -1, -2, 0 }) {
        int n = 3;
        GuardedVector<T> x = sample<T>(inc), x0 = x;
        double s = 0;
        int    imax = 0;
        double vmax = -1;
        for (int e=0; e<n; ++e) {
            double a = abs1d(x.elem(inc==0 ? 0 : e, n, inc));
            s += a;
            if (a>vmax) {
                vmax = a;
                imax = e;
            }
        }
        ctx.check(std::abs(ulm::asum(n, x.x(), inc)-s)<=1e-5*s
                  && x.sameAs(x0),
                  "ulmblas asum(n=%d, incX=%d), %s", n, inc, t);
        ctx.check(ulm::iamax(n, x.x(), inc)==imax && x.sameAs(x0),
                  "ulmblas i?amax(n=%d, incX=%d) is %d, expected %d, %s", n,
                  inc, ulm::iamax(n, x.x(), inc), imax, t);
        double r = ulm::nrm2(n, x.x(), inc), q = 0;
        for (int e=0; e<n; ++e) {
            double a = std::abs(toCdouble(x.elem(inc==0 ? 0 : e, n, inc)));
            q += a*a;
        }
        ctx.check(std::abs(r-std::sqrt(q))<=1e-5*std::sqrt(q)
                  && x.sameAs(x0),
                  "ulmblas nrm2(n=%d, incX=%d), %s", n, inc, t);
        if (inc!=0) {
            GuardedVector<T> y = x, expected = x;
            for (int e=0; e<n; ++e) {
                expected.elem(e, n, inc) *= T(2);
            }
            ulm::scal(n, T(2), y.x(), inc);
            ctx.check(y.sameAs(expected),
                      "ulmblas scal(n=%d, incX=%d), %s", n, inc, t);
        }
    }
    GuardedVector<T> x = sample<T>(1);
    ctx.check(ulm::iamax(0, x.x(), 1)==-1,
              "ulmblas i?amax(n=0) != -1, %s", t);
}

//
//  cblas_?her2k, row major: alpha in read-only memory (reference CBLAS
//  conjugates a copy).  A write is caught as SIGSEGV or SIGBUS.
//
sigjmp_buf  writeFault;

extern "C" void
onWriteFault(int)
{
    siglongjmp(writeFault, 1);
}

template <typename T>
bool
her2kReadOnlyAlpha()
{
    typedef typename Real<T>::type R;

    const long  page  = sysconf(_SC_PAGESIZE);
    void        *p    = mmap(0, page, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p==MAP_FAILED) {
        return false;
    }
    R *alpha = static_cast<R *>(p);
    alpha[0] = 1;
    alpha[1] = 2;
    mprotect(p, page, PROT_READ);

    std::vector<T> A(4, T(1)), B(4, T(2)), C(4, T(0));
    struct sigaction sa, oldSegv, oldBus;
    sa.sa_handler = onWriteFault;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, &oldSegv);
    sigaction(SIGBUS, &sa, &oldBus);

    bool ok = false;
    if (sigsetjmp(writeFault, 1)==0) {
        if (std::is_same<T, cfloat>::value) {
            cblas_cher2k(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2,
                         reinterpret_cast<const float *>(alpha),
                         reinterpret_cast<const float *>(&A[0]), 2,
                         reinterpret_cast<const float *>(&B[0]), 2, 1.0f,
                         reinterpret_cast<float *>(&C[0]), 2);
        } else {
            cblas_zher2k(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2,
                         reinterpret_cast<const double *>(alpha),
                         reinterpret_cast<const double *>(&A[0]), 2,
                         reinterpret_cast<const double *>(&B[0]), 2, 1.0,
                         reinterpret_cast<double *>(&C[0]), 2);
        }
        // C = alpha A B^H + conj(alpha) B A^H: alpha = (1,2), A = 1,
        // B = 2: (1+2i)*4 + (1-2i)*4 = 8 in the upper triangle (row
        // major: C[0], C[1], C[3]); the lower triangle C[2] is not touched
        ok = C[0]==T(8) && C[1]==T(8) && C[3]==T(8) && C[2]==T(0);
    }
    sigaction(SIGSEGV, &oldSegv, 0);
    sigaction(SIGBUS, &oldBus, 0);
    munmap(p, page);
    return ok;
}

//
//  ?rotg as in LAPACK 3.12.1 (values of the reference BLAS)
//
template <typename R>
bool
near(R x, R y, double tol)
{
    if (std::isnan(x) || std::isnan(y)) {
        return std::isnan(x) && std::isnan(y);
    }
    if (std::isinf(x) || std::isinf(y)) {
        return x==y;
    }
    return std::abs(double(x)-double(y))
        <= tol*std::max(std::abs(double(x)), std::abs(double(y)));
}

template <typename R>
void
testRealRotg(Context &ctx)
{
    const char *t   = typeName<R>();
    const R     inf = std::numeric_limits<R>::infinity();
    const R     big = std::numeric_limits<R>::max()/2;
    const double tol = 4*std::numeric_limits<R>::epsilon();

    struct Case { R a, b, r, z, c, s; };
    const R sqrt2 = std::sqrt(R(2)), rs2 = R(1)/sqrt2;
    const Case cases[] = {
        // a, b -> r, z, c, s
        { 3, 4, 5, R(5)/R(3), R(0.6), R(0.8) },
        { 4, 3, 5, R(0.6), R(0.8), R(0.6) },
        { -3, 4, 5, R(-5)/R(3), R(-0.6), R(0.8) },
        { big, big, big*sqrt2, sqrt2, rs2, rs2 },              // no overflow
        { -big, big, big*sqrt2, -sqrt2, -rs2, rs2 },
        { inf, 0, inf, 0, 1, 0 },
        { 0, inf, inf, 1, 0, 1 },
        { 0, -3, -3, 1, 0, 1 },
        { -3, 0, -3, 0, 1, 0 },
    };
    for (const Case &k : cases) {
        R a = k.a, b = k.b, c, s;
        blas::rotg(&a, &b, &c, &s);
        ctx.check(near(a, k.r, tol) && near(b, k.z, tol) && near(c, k.c, tol)
                  && near(s, k.s, tol),
                  "%srotg(%g, %g): r=%g z=%g c=%g s=%g, expected %g %g %g %g",
                  t, double(k.a), double(k.b), double(a), double(b),
                  double(c), double(s), double(k.r), double(k.z),
                  double(k.c), double(k.s));
    }
    // signs of zeros as in DROTG of LAPACK 3.12.1
    R a = 0, b = -3, c, s;
    blas::rotg(&a, &b, &c, &s);
    ctx.check(!std::signbit(c), "%srotg(0, -3): c is -0", t);
    a = -3; b = 0;
    blas::rotg(&a, &b, &c, &s);
    ctx.check(!std::signbit(s) && !std::signbit(b),
              "%srotg(-3, 0): s or z is -0", t);
}

template <typename R>
void
testComplexRotg(Context &ctx)
{
    typedef std::complex<R> C;

    const char  *t  = typeName<C>();
    const R     big = std::numeric_limits<R>::max()/2;
    const double tol = 8*std::numeric_limits<R>::epsilon();
    const R     rs2 = R(1)/std::sqrt(R(2));

    // a = b = (big, big): |a|^2 + |b|^2 overflows
    C a(big, big), b(big, big), s;
    R c;
    blas::rotg(&a, &b, &c, &s);
    ctx.check(near(a.real(), big*std::sqrt(R(2)), tol)
              && near(a.imag(), big*std::sqrt(R(2)), tol)
              && near(c, rs2, tol) && near(s.real(), rs2, tol)
              && std::abs(s.imag())<=tol,
              "%srotg((big,big), (big,big)): r=(%g,%g) c=%g s=(%g,%g)", t,
              double(a.real()), double(a.imag()), double(c),
              double(s.real()), double(s.imag()));
}

} // namespace

ULMBLAS_TEST(T0900, "conformance/cblas level 1 with increments <= 0")
{
    testLevel1Cblas<float>(ctx); testLevel1Cblas<double>(ctx);
    testLevel1Cblas<cfloat>(ctx); testLevel1Cblas<cdouble>(ctx);
}

ULMBLAS_TEST(T0901, "conformance/ulmblas level 1 with general increments")
{
    testLevel1Ulmblas<float>(ctx); testLevel1Ulmblas<double>(ctx);
    testLevel1Ulmblas<cfloat>(ctx); testLevel1Ulmblas<cdouble>(ctx);
}

ULMBLAS_TEST(T0902, "conformance/cblas_?her2k row major, read-only alpha")
{
    ctx.check(her2kReadOnlyAlpha<cfloat>(), "cblas_cher2k wrote to alpha "
              "or computed a wrong result");
    ctx.check(her2kReadOnlyAlpha<cdouble>(), "cblas_zher2k wrote to alpha "
              "or computed a wrong result");
}

ULMBLAS_TEST(T0903, "conformance/?rotg as in LAPACK 3.12.1")
{
    testRealRotg<float>(ctx); testRealRotg<double>(ctx);
    testComplexRotg<float>(ctx); testComplexRotg<double>(ctx);
}
