//
//  Tests of the template functions ulmBLAS::getrf and ulmBLAS::potrf for
//  float, double, complex<float> and complex<double>.
//
//  Every matrix lives in a buffer of its own and is described by a pointer
//  to element (0,0) and two strides (column major, row major, row stride 2,
//  neither stride 1, negative strides).  The other elements of the buffer
//  hold a marker and must not change (bitwise); for potrf also the triangle
//  that is not referenced.  Residuals are computed in complex<double>.
//
//  Built with OpenMP the larger cases run the task-based algorithms; the
//  sizes cross their thresholds (min(m,n) > 150 for getrf, n > 100 for
//  potrf).
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <ulmblas/ulmblas.h>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/potrf.h>
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

using namespace test;

namespace {

typedef std::int64_t Index;

template <typename T>
struct Real
{
    typedef T type;
};

template <typename T>
struct Real<std::complex<T> >
{
    typedef T type;
};

template <typename T>
double
eps()
{
    return std::numeric_limits<typename Real<T>::type>::epsilon();
}

template <typename T>
struct IsComplex
{
    static const bool value = false;
};

template <typename T>
struct IsComplex<std::complex<T> >
{
    static const bool value = true;
};

//
//  A strided matrix in a buffer of its own: element (i,j) is
//  data[off + i*rs + j*cs], the rest of the buffer holds a marker.
//
template <typename T>
struct Strided
{
    Strided(Index m_, Index n_, Index rs_, Index cs_)
        : m(m_), n(n_), rs(rs_), cs(cs_)
    {
        const Index lo = std::min<Index>(0, (m-1)*rs) + std::min<Index>(0, (n-1)*cs);
        const Index hi = std::max<Index>(0, (m-1)*rs) + std::max<Index>(0, (n-1)*cs);
        buf.assign(std::size_t(hi-lo+1), fromCdouble<T>(cdouble(1234.5, -77.25)));
        off = -lo;
    }

    T &
    operator()(Index i, Index j)
    {
        return buf[std::size_t(off+i*rs+j*cs)];
    }

    T *
    ptr()
    {
        return &buf[std::size_t(off)];
    }

    Index           m, n, rs, cs, off;
    std::vector<T>  buf;
};

//  layouts: (row stride, column stride) for an m x n matrix
std::vector<std::pair<Index, Index> >
layouts(Index m, Index n)
{
    std::vector<std::pair<Index, Index> > l;
    l.push_back(std::make_pair(Index(1), m+3));         // column major, ld > m
    l.push_back(std::make_pair(n+2, Index(1)));         // row major, ld > n
    l.push_back(std::make_pair(Index(2), 2*m+1));       // every second row
    l.push_back(std::make_pair(3*n+1, Index(3)));       // neither stride 1
    l.push_back(std::make_pair(Index(-2), -(2*m+5)));   // negative strides
    return l;
}

template <typename T>
bool
sameBits(const T &a, const T &b)
{
    return std::memcmp(&a, &b, sizeof(T))==0;
}

//
//  getrf: P A = L U.  Checks the residual, |L| <= 1 (complex: the pivot
//  has the largest |re| + |im|, so only |L| <= sqrt(2)), the pivots against expect (if not null), INFO, and that the
//  rest of the buffer is unchanged.
//
template <typename T>
bool
getrfCheck(Context &ctx, const char *what, const Full &A0, Strided<T> &A,
           const std::vector<Index> *expectPiv, Index expectInfo)
{
    const Index m = A.m, n = A.n, k = std::min(m, n);
    const Strided<T> before = A;
    std::vector<Index> piv(std::size_t(std::max<Index>(k, 1)), -1);

    const Index info = ulmBLAS::getrf(m, n, A.ptr(), A.rs, A.cs, piv.data(),
                                      Index(1));

    char tag[160];
    std::snprintf(tag, sizeof(tag), "%s m=%lld n=%lld rs=%lld cs=%lld", what,
                  (long long)m, (long long)n, (long long)A.rs, (long long)A.cs);

    if (!ctx.check(info==expectInfo, "getrf %s: info %lld, expected %lld",
                   tag, (long long)info, (long long)expectInfo))
    {
        return false;
    }
    for (Index i=0; i<k; ++i) {
        if (!ctx.check(piv[i]>=i && piv[i]<m, "getrf %s: piv[%lld] = %lld",
                       tag, (long long)i, (long long)piv[i]))
        {
            return false;
        }
        if (expectPiv && !ctx.check(piv[i]==(*expectPiv)[i],
                                    "getrf %s: piv[%lld] = %lld, expected %lld",
                                    tag, (long long)i, (long long)piv[i],
                                    (long long)(*expectPiv)[i]))
        {
            return false;
        }
    }
    //  the rest of the buffer
    std::vector<char> inside(A.buf.size(), 0);
    for (Index j=0; j<n; ++j) {
        for (Index i=0; i<m; ++i) {
            inside[std::size_t(A.off+i*A.rs+j*A.cs)] = 1;
        }
    }
    for (std::size_t q=0; q<A.buf.size(); ++q) {
        if (!inside[q] && !ctx.check(sameBits(A.buf[q], before.buf[q]),
                                     "getrf %s: buffer element %zu outside "
                                     "of A changed", tag, q))
        {
            return false;
        }
    }
    //  |L| <= 1 (sqrt(2) for complex)
    const double bound = IsComplex<T>::value ? std::sqrt(2.0)*(1+4*eps<T>()) : 1.0;
    for (Index j=0; j<k; ++j) {
        for (Index i=j+1; i<m; ++i) {
            if (!ctx.check(std::abs(toCdouble(A(i,j)))<=bound,
                           "getrf %s: |L(%lld,%lld)| > 1", tag, (long long)i,
                           (long long)j))
            {
                return false;
            }
        }
    }
    //  residual: L U and P A0
    Full PA = A0;
    for (Index i=0; i<k; ++i) {
        if (piv[i]!=i) {
            for (Index j=0; j<n; ++j) {
                std::swap(PA(int(i), int(j)), PA(int(piv[i]), int(j)));
            }
        }
    }
    double norm = 0, err = 0;
    for (Index i=0; i<m; ++i) {
        for (Index j=0; j<n; ++j) {
            cdouble s = 0;
            for (Index l=0; l<=std::min(i, std::min(j, k-1)); ++l) {
                const cdouble lil = (l==i) ? cdouble(1) : toCdouble(A(i,l));
                s += lil*toCdouble(A(l,j));
            }
            err  = std::max(err, std::abs(s-PA(int(i), int(j))));
            norm = std::max(norm, std::abs(PA(int(i), int(j))));
        }
    }
    const double ratio = err/(std::max(norm, 1e-300)*double(std::max(m, n))*eps<T>());
    return ctx.check(ratio<30, "getrf %s: residual ratio %.3g", tag, ratio);
}

template <typename T>
void
fill(Full &A0, Strided<T> &A, Random &random)
{
    for (Index j=0; j<A.n; ++j) {
        for (Index i=0; i<A.m; ++i) {
            const T x = random.value<T>();
            A(i,j) = x;
            A0(int(i), int(j)) = toCdouble(x);
        }
    }
}

template <typename T>
void
testGetrfRandom(Context &ctx)
{
    Random random(0x4c55);
    const Index size[][2] = { {1, 1}, {1, 7}, {7, 1}, {5, 3}, {3, 5},
                              {33, 33}, {97, 80}, {80, 97}, {170, 170},
                              {260, 200}, {200, 260}, {420, 420} };
    for (const auto &s : size) {
        for (const auto &l : layouts(s[0], s[1])) {
            Full       A0{int(s[0]), int(s[1])};
            Strided<T> A(s[0], s[1], l.first, l.second);
            fill(A0, A, random);
            if (!getrfCheck(ctx, "random", A0, A, 0, 0)) {
                return;
            }
        }
    }
}

//
//  A = P0 L0 U0 with |L0(i,j)| <= 1/2 below the diagonal (complex: |re| +
//  |im| <= 1/2): the pivots of partial pivoting are the interchanges of P0
//  (exactly, no ties).
//
template <typename T>
void
testGetrfKnownPivots(Context &ctx)
{
    Random random(0x9e37);
    const Index size[] = { 6, 40, 171, 300 };
    for (Index n : size) {
        std::vector<Index> p0(static_cast<std::size_t>(n));
        for (Index i=0; i<n; ++i) {
            p0[i] = i + Index((random.uniform()+1)/2*double(n-i));
            p0[i] = std::min(p0[i], n-1);
        }
        Full L{int(n), int(n)}, U{int(n), int(n)};
        for (Index j=0; j<n; ++j) {
            L(int(j), int(j)) = 1;
            for (Index i=j+1; i<n; ++i) {
                L(int(i), int(j)) = toCdouble(random.value<T>())
                                  * (IsComplex<T>::value ? 0.25 : 0.5);
            }
            for (Index i=0; i<=j; ++i) {
                U(int(i), int(j)) = toCdouble(random.value<T>());
            }
            //  diagonally dominant: U is well conditioned, so the rounding
            //  of A to T does not change the pivots
            U(int(j), int(j)) += (U(int(j), int(j)).real()>=0) ? double(n)
                                                               : -double(n);
        }
        for (const auto &l : layouts(n, n)) {
            Full       A0{int(n), int(n)};
            Strided<T> A(n, n, l.first, l.second);
            //  A0 = L U (rounded to T), then the interchanges of P0 in
            //  reverse order: P0 A0 = L U
            for (Index i=0; i<n; ++i) {
                for (Index j=0; j<n; ++j) {
                    cdouble s = 0;
                    for (Index q=0; q<=std::min(i, j); ++q) {
                        s += L(int(i), int(q))*U(int(q), int(j));
                    }
                    A0(int(i), int(j)) = toCdouble(fromCdouble<T>(s));
                }
            }
            for (Index i=n-1; i>=0; --i) {
                for (Index j=0; j<n; ++j) {
                    std::swap(A0(int(i), int(j)), A0(int(p0[i]), int(j)));
                }
            }
            for (Index j=0; j<n; ++j) {
                for (Index i=0; i<n; ++i) {
                    A(i,j) = fromCdouble<T>(A0(int(i), int(j)));
                }
            }
            if (!getrfCheck(ctx, "P0 L0 U0", A0, A, &p0, 0)) {
                return;
            }
        }
    }
}

//  zero column j: U(j,j) is the first zero pivot, INFO = j+1
template <typename T>
void
testGetrfSingular(Context &ctx)
{
    Random random(0x51e7);
    const Index cases[][3] = { {5, 5, 0}, {5, 5, 4}, {40, 30, 17},
                               {200, 200, 3}, {200, 200, 150}, {300, 260, 259} };
    for (const auto &c : cases) {
        for (const auto &l : layouts(c[0], c[1])) {
            Full       A0{int(c[0]), int(c[1])};
            Strided<T> A(c[0], c[1], l.first, l.second);
            fill(A0, A, random);
            for (Index i=0; i<c[0]; ++i) {
                A(i, c[2]) = T(0);
                A0(int(i), int(c[2])) = 0;
            }
            if (!getrfCheck(ctx, "zero column", A0, A, 0, c[2]+1)) {
                return;
            }
        }
    }
}

//
//  potrf: A = L L^H (lower) or U^H U (upper).  A Hermitian positive
//  definite matrix (random, n added to the diagonal); bad >= 0 makes the
//  leading minor of order bad+1 indefinite (INFO = bad+1).
//
template <typename T>
bool
potrfCase(Context &ctx, Random &random, bool lower, Index n, Index rs, Index cs,
          Index bad)
{
    Full       F{int(n), int(n)};
    Strided<T> A(n, n, rs, cs);
    for (Index j=0; j<n; ++j) {
        for (Index i=j; i<n; ++i) {
            cdouble x = toCdouble(random.value<T>());
            if (i==j) {
                x = cdouble(x.real()+double(n), 0);
            }
            F(int(i), int(j)) = x;
            F(int(j), int(i)) = std::conj(x);
        }
    }
    if (bad>=0) {
        F(int(bad), int(bad)) = -double(n)*double(n);
    }
    for (Index j=0; j<n; ++j) {
        for (Index i=0; i<n; ++i) {
            if (lower ? i>=j : i<=j) {
                A(i,j) = fromCdouble<T>(F(int(i), int(j)));
            }
        }
    }
    const Strided<T> before = A;
    const Index info = ulmBLAS::potrf(lower, n, A.ptr(), rs, cs);

    char tag[160];
    std::snprintf(tag, sizeof(tag), "%s n=%lld rs=%lld cs=%lld",
                  lower ? "lower" : "upper", (long long)n, (long long)rs,
                  (long long)cs);
    const Index expect = (bad>=0) ? bad+1 : 0;
    if (!ctx.check(info==expect, "potrf %s: info %lld, expected %lld", tag,
                   (long long)info, (long long)expect))
    {
        return false;
    }
    //  the other triangle and the rest of the buffer
    std::vector<char> inside(A.buf.size(), 0);
    for (Index j=0; j<n; ++j) {
        for (Index i=0; i<n; ++i) {
            if (lower ? i>=j : i<=j) {
                inside[std::size_t(A.off+i*rs+j*cs)] = 1;
            }
        }
    }
    for (std::size_t q=0; q<A.buf.size(); ++q) {
        if (!inside[q] && !ctx.check(sameBits(A.buf[q], before.buf[q]),
                                     "potrf %s: element %zu outside of the "
                                     "triangle changed", tag, q))
        {
            return false;
        }
    }
    if (bad>=0) {
        return true;
    }
    //  L (lower: L(i,j) = A(i,j); upper: L = U^H)
    Full L{int(n), int(n)};
    for (Index j=0; j<n; ++j) {
        if (!ctx.check(toCdouble(A(j,j)).imag()==0 && toCdouble(A(j,j)).real()>0,
                       "potrf %s: diagonal element %lld not real positive", tag,
                       (long long)j))
        {
            return false;
        }
        for (Index i=j; i<n; ++i) {
            L(int(i), int(j)) = lower ? toCdouble(A(i,j))
                                      : std::conj(toCdouble(A(j,i)));
        }
    }
    double norm = 0, err = 0;
    for (Index i=0; i<n; ++i) {
        for (Index j=0; j<n; ++j) {
            cdouble s = 0;
            for (Index l=0; l<=std::min(i, j); ++l) {
                s += L(int(i), int(l))*std::conj(L(int(j), int(l)));
            }
            err  = std::max(err, std::abs(s-F(int(i), int(j))));
            norm = std::max(norm, std::abs(F(int(i), int(j))));
        }
    }
    const double ratio = err/(norm*double(n)*eps<T>());
    return ctx.check(ratio<30, "potrf %s: residual ratio %.3g", tag, ratio);
}

template <typename T>
void
testPotrf(Context &ctx)
{
    Random random(0x9071);
    const Index size[] = { 1, 2, 7, 64, 65, 101, 180, 333 };
    for (Index n : size) {
        for (int lower=0; lower<2; ++lower) {
            for (const auto &l : layouts(n, n)) {
                if (!potrfCase<T>(ctx, random, lower!=0, n, l.first, l.second, -1)) {
                    return;
                }
            }
        }
    }
}

template <typename T>
void
testPotrfNotDefinite(Context &ctx)
{
    Random random(0x1d3f);
    const Index cases[][2] = { {1, 0}, {5, 0}, {5, 4}, {90, 45},
                               {200, 0}, {200, 130}, {333, 332} };
    for (const auto &c : cases) {
        for (int lower=0; lower<2; ++lower) {
            for (const auto &l : layouts(c[0], c[0])) {
                if (!potrfCase<T>(ctx, random, lower!=0, c[0], l.first, l.second,
                                  c[1]))
                {
                    return;
                }
            }
        }
    }
}

} // namespace

ULMBLAS_TEST(T0410, "lapack/ulmBLAS::getrf, random, strides")
{
    testGetrfRandom<float>(ctx);
    testGetrfRandom<double>(ctx);
    testGetrfRandom<cfloat>(ctx);
    testGetrfRandom<cdouble>(ctx);
}

ULMBLAS_TEST(T0411, "lapack/ulmBLAS::getrf, known pivots (P0 L0 U0)")
{
    testGetrfKnownPivots<float>(ctx);
    testGetrfKnownPivots<double>(ctx);
    testGetrfKnownPivots<cfloat>(ctx);
    testGetrfKnownPivots<cdouble>(ctx);
}

ULMBLAS_TEST(T0412, "lapack/ulmBLAS::getrf, zero column (INFO)")
{
    testGetrfSingular<float>(ctx);
    testGetrfSingular<double>(ctx);
    testGetrfSingular<cfloat>(ctx);
    testGetrfSingular<cdouble>(ctx);
}

ULMBLAS_TEST(T0413, "lapack/ulmBLAS::potrf, lower and upper, strides")
{
    testPotrf<float>(ctx);
    testPotrf<double>(ctx);
    testPotrf<cfloat>(ctx);
    testPotrf<cdouble>(ctx);
}

ULMBLAS_TEST(T0414, "lapack/ulmBLAS::potrf, not positive definite (INFO)")
{
    testPotrfNotDefinite<float>(ctx);
    testPotrfNotDefinite<double>(ctx);
    testPotrfNotDefinite<cfloat>(ctx);
    testPotrfNotDefinite<cdouble>(ctx);
}
