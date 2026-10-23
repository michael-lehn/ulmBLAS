#ifndef ULMBLAS_TEST_NUMERIC_TESTING_H
#define ULMBLAS_TEST_NUMERIC_TESTING_H 1

//
//  Minimal test framework for the numerical tests of ulmBLAS.
//
//  A test case has a stable ID (Tnnnn, never reused, see
//  test/known_issues.md) and is defined by
//
//      ULMBLAS_TEST(T0123, "level3/dgemm: description")
//      {
//          ...
//          ctx.check(condition, "format", ...);
//      }
//
//  A test case covering a known bug (an OPEN issue #nnnn of type BUG in
//  test/known_issues.md) is defined with ULMBLAS_TEST_XFAIL(ID, NAME, nnnn).
//  It is expected to fail; if it passes it is reported as XPASS, which is an
//  error: the issue has to be resolved and the test turned into a regular one.
//
//  A test case concerning an open design question (an OPEN issue of type
//  DESIGN) records the observed behavior instead of asserting one:
//
//      ULMBLAS_TEST_DESIGN(T0123, "name", nnnn, "recorded behavior")
//      {
//          ctx.observed = ...;
//      }
//
//  It is reported as DESIGN if ctx.observed equals the recorded behavior and
//  as FAIL otherwise (then the registry and the test have to be updated).
//

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace test {

typedef std::complex<float>   cfloat;
typedef std::complex<double>  cdouble;

//
//  Scalar type traits
//
template <typename T> struct Real                 { typedef T type; };
template <typename T> struct Real<std::complex<T> > { typedef T type; };

template <typename T> struct IsComplex                    { enum { value = 0 }; };
template <typename T> struct IsComplex<std::complex<T> >  { enum { value = 1 }; };

template <typename T>
const char *
typeName()
{
    return std::is_same<T,float>::value  ? "float"
         : std::is_same<T,double>::value ? "double"
         : std::is_same<T,cfloat>::value ? "complex<float>"
         : "complex<double>";
}

template <typename T>
typename Real<T>::type
eps()
{
    return std::numeric_limits<typename Real<T>::type>::epsilon();
}

inline float   conjugate(float x)          { return x; }
inline double  conjugate(double x)         { return x; }
inline cfloat  conjugate(const cfloat &x)  { return std::conj(x); }
inline cdouble conjugate(const cdouble &x) { return std::conj(x); }

inline float   realPart(float x)           { return x; }
inline double  realPart(double x)          { return x; }
inline float   realPart(const cfloat &x)   { return x.real(); }
inline double  realPart(const cdouble &x)  { return x.real(); }

template <typename T>
bool
isFinite(const T &x)
{
    return std::isfinite(std::real(x)) && std::isfinite(std::imag(x));
}

//
//  Bitwise comparison (NaNs compare equal to themselves).
//
template <typename T>
bool
sameBits(const T &a, const T &b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <typename T>
T
nan()
{
    typedef typename Real<T>::type R;
    return T(std::numeric_limits<R>::quiet_NaN());
}

//
//  Deterministic pseudo random numbers (xorshift64*), uniform in [-1,1].
//  Results do not depend on the C++ standard library.
//
class Random
{
    public:
        explicit
        Random(unsigned long long seed = 0x2545F4914F6CDD1DULL)
            : state_(seed ? seed : 1)
        {
        }

        double
        uniform()
        {
            state_ ^= state_ >> 12;
            state_ ^= state_ << 25;
            state_ ^= state_ >> 27;
            unsigned long long r = state_ * 0x2545F4914F6CDD1DULL;
            return 2.0*double(r >> 11)/double(1ULL << 53) - 1.0;
        }

        template <typename T>
        T
        value()
        {
            return make(static_cast<T *>(0));
        }

    private:
        float   make(float *)   { return float(uniform()); }
        double  make(double *)  { return uniform(); }
        cfloat  make(cfloat *)  { float re = float(uniform());
                                  return cfloat(re, float(uniform())); }
        cdouble make(cdouble *) { double re = uniform();
                                  return cdouble(re, uniform()); }

        unsigned long long  state_;
};

//
//  Test context: collects the results of the checks of one test case.
//
class Context
{
    public:
        Context()
            : numChecks(0), numFailed(0)
        {
        }

        // record a successful check
        void
        pass()
        {
            ++numChecks;
        }

        bool
        check(bool ok, const char *fmt, ...)
#if defined(__GNUC__)
            __attribute__((format(printf, 3, 4)))
#endif
        {
            ++numChecks;
            if (!ok) {
                ++numFailed;
                if (messages.size()<5) {
                    char     buffer[512];
                    va_list  ap;
                    va_start(ap, fmt);
                    std::vsnprintf(buffer, sizeof(buffer), fmt, ap);
                    va_end(ap);
                    messages.push_back(buffer);
                }
            }
            return ok;
        }

        long                      numChecks;
        long                      numFailed;
        std::vector<std::string>  messages;
        std::string               observed;     // design tests
};

//
//  Registry of test cases
//
typedef void (*TestFunction)(Context &);

struct TestCase
{
    const char    *id;
    const char    *name;
    TestFunction  function;
    int           issue;        // 0: regular test, else XFAIL or DESIGN
    const char    *recorded;    // design tests: recorded behavior, else 0
};

inline std::vector<TestCase> &
registry()
{
    static std::vector<TestCase> testCases;
    return testCases;
}

struct Registrar
{
    Registrar(const char *id, const char *name, TestFunction function,
              int issue, const char *recorded)
    {
        TestCase testCase = { id, name, function, issue, recorded };
        registry().push_back(testCase);
    }
};

#define ULMBLAS_TEST_DESIGN(ID, NAME, ISSUE, RECORDED)                      \
    static void test_##ID(test::Context &);                                 \
    static test::Registrar registrar_##ID(#ID, NAME, test_##ID, ISSUE,      \
                                          RECORDED);                        \
    static void test_##ID(test::Context &ctx)

#define ULMBLAS_TEST_XFAIL(ID, NAME, ISSUE)                                 \
    ULMBLAS_TEST_DESIGN(ID, NAME, ISSUE, 0)

#define ULMBLAS_TEST(ID, NAME)  ULMBLAS_TEST_XFAIL(ID, NAME, 0)

//
//  Strided matrix (column major or row major) with leading dimension larger
//  than necessary.  The padding and elements outside the region a function
//  may write are compared bitwise before and after the call.
//
template <typename T>
struct Matrix
{
    Matrix(int m_, int n_, bool rowMajor_, int extraLd = 3)
        : m(m_), n(n_), rowMajor(rowMajor_),
          ld(std::max(1, (rowMajor_ ? n_ : m_) + extraLd)),
          data(std::max(1, ld*(rowMajor_ ? m_ : n_)), T(-777))
    {
    }

    T &
    operator()(int i, int j)
    {
        return rowMajor ? data[i*ld+j] : data[i+j*ld];
    }

    const T &
    operator()(int i, int j) const
    {
        return rowMajor ? data[i*ld+j] : data[i+j*ld];
    }

    void
    randomize(Random &random)
    {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                (*this)(i,j) = random.value<T>();
            }
        }
    }

    // Bitwise compare everything that is not inside the m x n matrix
    // (padding) with a copy taken before the call.
    bool
    paddingUnchanged(const Matrix &before) const
    {
        for (std::size_t p=0; p<data.size(); ++p) {
            int  outer = int(p) / ld;
            int  inner = int(p) % ld;
            bool inside = rowMajor ? (outer<m && inner<n)
                                   : (outer<n && inner<m);
            if (!inside && !sameBits(data[p], before.data[p])) {
                return false;
            }
        }
        return true;
    }

    T       *ptr()       { return &data[0]; }
    const T *ptr() const { return &data[0]; }

    int             m, n;
    bool            rowMajor;
    int             ld;
    std::vector<T>  data;
};

//
//  Strided vector with increment inc (possibly negative).  As in BLAS, for
//  inc<0 element i is stored at position (1-n)*inc + i*inc.  Elements in the
//  gaps are compared bitwise before and after a call.
//
template <typename T>
struct Vector
{
    Vector(int n_, int inc_)
        : n(n_), inc(inc_),
          data(std::max(1, 1 + (n_>0 ? (n_-1)*std::abs(inc_) : 0)), T(-555))
    {
    }

    T &
    operator()(int i)
    {
        return data[inc>0 ? i*inc : (i-n+1)*inc];
    }

    const T &
    operator()(int i) const
    {
        return data[inc>0 ? i*inc : (i-n+1)*inc];
    }

    void
    randomize(Random &random)
    {
        for (int i=0; i<n; ++i) {
            (*this)(i) = random.value<T>();
        }
    }

    bool
    gapsUnchanged(const Vector &before) const
    {
        int a = std::abs(inc);
        for (std::size_t p=0; p<data.size(); ++p) {
            if ((n==0 || int(p)%a!=0) && !sameBits(data[p], before.data[p])) {
                return false;
            }
        }
        return true;
    }

    T       *ptr()       { return &data[0]; }
    const T *ptr() const { return &data[0]; }

    int             n, inc;
    std::vector<T>  data;
};

//
//  Error check of a computed value against a reference value.
//
//  bound is a bound on sum |terms| of the computed quantity (e.g. |alpha| *
//  sum_l |a_il| |b_lj| + |beta| |c_ij| for GEMM), k the number of terms.
//  Both the tested and the reference result have a rounding error of at
//  most about k*eps*bound (standard error analysis of inner products), so
//
//      |computed - reference| <= factor * (k+2) * eps * bound
//
//  is required, with factor 4 (real) or 8 (complex arithmetic).  The tests
//  do not require bitwise equality with any other BLAS implementation.
//
template <typename T>
double
tolerance(int k)
{
    return (IsComplex<T>::value ? 8.0 : 4.0) * (k+2) * double(eps<T>());
}

template <typename T>
bool
closeEnough(const T &computed, const T &reference, double bound, int k)
{
    if (!isFinite(computed)) {
        return false;
    }
    double err = std::abs(std::complex<double>(std::real(computed),
                                               std::imag(computed))
                        - std::complex<double>(std::real(reference),
                                               std::imag(reference)));
    double tiny = double(std::numeric_limits<typename Real<T>::type>::min());
    return err <= tolerance<T>(k)*bound + tiny;
}

template <typename T>
double
absValue(const T &x)
{
    return std::abs(std::complex<double>(std::real(x), std::imag(x)));
}

} // namespace test

#endif // ULMBLAS_TEST_NUMERIC_TESTING_H
