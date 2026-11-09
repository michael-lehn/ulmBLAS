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
//  T0415-T0417 call the C interface (ULMLAPACK(?getrf), ULMLAPACK(?potrf))
//  with column- and row-major matrices and invalid arguments.
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
#include LAPACK_HEADER
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>
#include <test/numeric/lapacktest.h>

using namespace test;

namespace {

using namespace test::lapacktest;

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

//
//  The C interface for each type
//
typedef ULMBLAS_INDEX_TYPE CIndex;

CIndex cGetrf(CBLAS_ORDER o, CIndex m, CIndex n, float *A, CIndex ld, CIndex *p)
{ return ULMLAPACK(sgetrf)(o, m, n, A, ld, p); }
CIndex cGetrf(CBLAS_ORDER o, CIndex m, CIndex n, double *A, CIndex ld, CIndex *p)
{ return ULMLAPACK(dgetrf)(o, m, n, A, ld, p); }
CIndex cGetrf(CBLAS_ORDER o, CIndex m, CIndex n, cfloat *A, CIndex ld, CIndex *p)
{ return ULMLAPACK(cgetrf)(o, m, n, A, ld, p); }
CIndex cGetrf(CBLAS_ORDER o, CIndex m, CIndex n, cdouble *A, CIndex ld, CIndex *p)
{ return ULMLAPACK(zgetrf)(o, m, n, A, ld, p); }

CIndex cPotrf(CBLAS_ORDER o, CBLAS_UPLO u, CIndex n, float *A, CIndex ld)
{ return ULMLAPACK(spotrf)(o, u, n, A, ld); }
CIndex cPotrf(CBLAS_ORDER o, CBLAS_UPLO u, CIndex n, double *A, CIndex ld)
{ return ULMLAPACK(dpotrf)(o, u, n, A, ld); }
CIndex cPotrf(CBLAS_ORDER o, CBLAS_UPLO u, CIndex n, cfloat *A, CIndex ld)
{ return ULMLAPACK(cpotrf)(o, u, n, A, ld); }
CIndex cPotrf(CBLAS_ORDER o, CBLAS_UPLO u, CIndex n, cdouble *A, CIndex ld)
{ return ULMLAPACK(zpotrf)(o, u, n, A, ld); }

//  ?getrf, column major: the same pivots and factors as ulmBLAS::getrf on
//  a copy (bitwise; without a team of threads both run the same code)
template <typename T>
void
testGetrfC(Context &ctx)
{
    Random random(0xc6e7);
    const CIndex size[][2] = { {1, 1}, {9, 4}, {4, 9}, {60, 60}, {130, 101} };
    for (const auto &s : size) {
        const CIndex m = s[0], n = s[1], ld = m+2, k = std::min(m, n);
        //  Both calls use the same storage (the SSE dot kernels sum in an
        //  order that depends on the alignment).  The values are compared
        //  with a tolerance: with -mfma the compiler may contract a*b+c in
        //  the library and in this test in different places.
        std::vector<T> A(static_cast<std::size_t>(ld*n)), A0, B;
        for (T &x : A) x = random.value<T>();
        A0 = A;
        std::vector<CIndex> pa(static_cast<std::size_t>(k)), pb(static_cast<std::size_t>(k));
        const Index  ib = ulmBLAS::getrf(Index(m), Index(n), A.data(), Index(1),
                                         Index(ld), pb.data(), Index(1));
        B = A;
        A = A0;
        const CIndex ia = cGetrf(CblasColMajor, m, n, A.data(), ld, pa.data());
        const double tol = 1000.0 * double(k) * double(eps<T>());
        bool same = ia==CIndex(ib) && pa==pb;
        for (std::size_t q=0; same && q<A.size(); ++q) {
            const cdouble a = toCdouble(A[q]), b = toCdouble(B[q]);
            same = std::abs(a-b) <= tol * (1.0 + std::abs(b));
        }
        if (!ctx.check(same, "C ?getrf m=%d n=%d: differs from ulmBLAS::getrf",
                       int(m), int(n)))
        {
            return;
        }
    }
}

//  ?potrf, column and row major, lower and upper: residual (as potrfCase)
template <typename T>
void
testPotrfC(Context &ctx)
{
    Random random(0xc907);
    const CIndex size[] = { 1, 5, 64, 120 };
    for (CIndex n : size) {
        for (int row=0; row<2; ++row) {
            for (int lower=0; lower<2; ++lower) {
                const CIndex ld = n+3;
                //  A(i,j) = data[i*rs + j*cs]
                const Index rs = row ? Index(ld) : Index(1);
                const Index cs = row ? Index(1) : Index(ld);
                std::vector<T> data(static_cast<std::size_t>(ld*n), T(0));
                Full F{int(n), int(n)};
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
                for (Index j=0; j<n; ++j) {
                    for (Index i=0; i<n; ++i) {
                        data[std::size_t(i*rs+j*cs)]
                            = fromCdouble<T>(F(int(i), int(j)));
                    }
                }
                const CIndex info = cPotrf(row ? CblasRowMajor : CblasColMajor,
                                           lower ? CblasLower : CblasUpper,
                                           n, data.data(), ld);
                if (!ctx.check(info==0, "C ?potrf n=%d: info %d", int(n),
                               int(info)))
                {
                    return;
                }
                double norm = 0, err = 0;
                for (Index i=0; i<n; ++i) {
                    for (Index j=0; j<n; ++j) {
                        cdouble sum = 0;
                        for (Index l=0; l<=std::min(i, j); ++l) {
                            //  lower: L(i,l) = A(i,l); upper: L = U^H
                            const cdouble li = lower
                                ? toCdouble(data[std::size_t(i*rs+l*cs)])
                                : std::conj(toCdouble(data[std::size_t(l*rs+i*cs)]));
                            const cdouble lj = lower
                                ? toCdouble(data[std::size_t(j*rs+l*cs)])
                                : std::conj(toCdouble(data[std::size_t(l*rs+j*cs)]));
                            sum += li*std::conj(lj);
                        }
                        err  = std::max(err, std::abs(sum-F(int(i), int(j))));
                        norm = std::max(norm, std::abs(F(int(i), int(j))));
                    }
                }
                const double ratio = err/(norm*double(n)*eps<T>());
                if (!ctx.check(ratio<30, "C ?potrf n=%d %s %s: residual "
                               "ratio %.3g", int(n), row ? "row" : "col",
                               lower ? "lower" : "upper", ratio))
                {
                    return;
                }
            }
        }
    }
}

//  invalid arguments: -i for argument i, A not referenced
template <typename T>
void
testArgsC(Context &ctx)
{
    T      A[4] = { T(1), T(2), T(3), T(4) };
    CIndex p[2];
    const CBLAS_ORDER bad = CBLAS_ORDER(42);
    ctx.check(cGetrf(bad, 2, 2, A, 2, p)==-1, "C ?getrf: order");
    ctx.check(cGetrf(CblasColMajor, -1, 2, A, 2, p)==-2, "C ?getrf: m < 0");
    ctx.check(cGetrf(CblasColMajor, 2, -1, A, 2, p)==-3, "C ?getrf: n < 0");
    ctx.check(cGetrf(CblasColMajor, 2, 2, A, 1, p)==-5, "C ?getrf: ldA < m");
    ctx.check(cGetrf(CblasRowMajor, 1, 2, A, 1, p)==-5, "C ?getrf: ldA < n");
    ctx.check(cPotrf(bad, CblasLower, 2, A, 2)==-1, "C ?potrf: order");
    ctx.check(cPotrf(CblasColMajor, CBLAS_UPLO(42), 2, A, 2)==-2,
              "C ?potrf: upLo");
    ctx.check(cPotrf(CblasColMajor, CblasLower, -1, A, 2)==-3, "C ?potrf: n < 0");
    ctx.check(cPotrf(CblasColMajor, CblasLower, 2, A, 1)==-5, "C ?potrf: ldA < n");
    ctx.check(cGetrf(CblasColMajor, 0, 2, A, 1, p)==0
              && cPotrf(CblasColMajor, CblasLower, 0, A, 1)==0,
              "C: m = 0 or n = 0 is valid");
    const T B[4] = { T(1), T(2), T(3), T(4) };
    bool same = true;
    for (int i=0; i<4; ++i) {
        same = same && sameBits(A[i], B[i]);
    }
    ctx.check(same, "C: A changed by a call with invalid arguments");
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

ULMBLAS_TEST(T0415, "lapack/C interface ?getrf, column major")
{
    testGetrfC<float>(ctx);
    testGetrfC<double>(ctx);
    testGetrfC<cfloat>(ctx);
    testGetrfC<cdouble>(ctx);
}

ULMBLAS_TEST(T0416, "lapack/C interface ?potrf, column and row major")
{
    testPotrfC<float>(ctx);
    testPotrfC<double>(ctx);
    testPotrfC<cfloat>(ctx);
    testPotrfC<cdouble>(ctx);
}

ULMBLAS_TEST(T0417, "lapack/C interface ?getrf, ?potrf: invalid arguments")
{
    testArgsC<float>(ctx);
    testArgsC<double>(ctx);
    testArgsC<cfloat>(ctx);
    testArgsC<cdouble>(ctx);
}
