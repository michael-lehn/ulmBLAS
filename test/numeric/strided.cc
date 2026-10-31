//
//  GEMM with arbitrary strides (ulmblas_?gemm_strided, ulmblas_strided.h).
//
//  Every operand lives in a buffer of its own and is described by a
//  pointer to element (0,0) and two strides (column major, row major,
//  non-unit, negative, mixed, zero).  The result is compared with a
//  reference in complex<double>; the elements of the buffer of C that are
//  not elements of C must not change (bitwise).
//
#include <cstdint>
#include <vector>
#include <ulmblas_strided.h>
#include <test/numeric/reference.h>

using namespace test;

namespace {

//
//  The C functions for each type
//
void
gemmStrided(std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
            bool, const float *A, std::int64_t rA, std::int64_t cA,
            bool, const float *B, std::int64_t rB, std::int64_t cB,
            float beta, float *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_sgemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB,
                          beta, C, rC, cC);
}

void
gemmStrided(std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
            bool, const double *A, std::int64_t rA, std::int64_t cA,
            bool, const double *B, std::int64_t rB, std::int64_t cB,
            double beta, double *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_dgemm_strided(m, n, k, alpha, A, rA, cA, B, rB, cB,
                          beta, C, rC, cC);
}

void
gemmStrided(std::int64_t m, std::int64_t n, std::int64_t k,
            const cfloat &alpha, bool conjA, const cfloat *A,
            std::int64_t rA, std::int64_t cA, bool conjB, const cfloat *B,
            std::int64_t rB, std::int64_t cB, const cfloat &beta,
            cfloat *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_cgemm_strided(m, n, k, reinterpret_cast<const float *>(&alpha),
                          conjA, reinterpret_cast<const float *>(A), rA, cA,
                          conjB, reinterpret_cast<const float *>(B), rB, cB,
                          reinterpret_cast<const float *>(&beta),
                          reinterpret_cast<float *>(C), rC, cC);
}

void
gemmStrided(std::int64_t m, std::int64_t n, std::int64_t k,
            const cdouble &alpha, bool conjA, const cdouble *A,
            std::int64_t rA, std::int64_t cA, bool conjB, const cdouble *B,
            std::int64_t rB, std::int64_t cB, const cdouble &beta,
            cdouble *C, std::int64_t rC, std::int64_t cC)
{
    ulmblas_zgemm_strided(m, n, k, reinterpret_cast<const double *>(&alpha),
                          conjA, reinterpret_cast<const double *>(A), rA, cA,
                          conjB, reinterpret_cast<const double *>(B), rB, cB,
                          reinterpret_cast<const double *>(&beta),
                          reinterpret_cast<double *>(C), rC, cC);
}

//
//  Layouts of an r x c operand.  ZeroRow and ZeroCol (stride 0) only for
//  A and B, or for C with r = 1 (c = 1).
//
enum Layout { ColMajor, RowMajor, RowStride2, NegRow, NegBoth, Mixed,
              ZeroRow, ZeroCol, NumLayouts };

const char *
layoutName(Layout l)
{
    static const char *name[] = { "col", "row", "rowstride2", "negrow",
                                  "negboth", "mixed", "zerorow", "zerocol" };
    return name[l];
}

template <typename T>
struct Operand
{
    Operand(Layout layout, std::int64_t r_, std::int64_t c_)
        : r(r_), c(c_)
    {
        std::int64_t R = std::max<std::int64_t>(r, 1);
        std::int64_t C = std::max<std::int64_t>(c, 1);
        switch (layout) {
            case ColMajor:   rs = 1;      cs = R+3;        break;
            case RowMajor:   rs = C+2;    cs = 1;          break;
            case RowStride2: rs = 2;      cs = 2*R+1;      break;
            case NegRow:     rs = -1;     cs = R+1;        break;
            case NegBoth:    rs = -C;     cs = -1;         break;
            case Mixed:      rs = 3;      cs = -(3*R+2);   break;
            case ZeroRow:    rs = 0;      cs = 2;          break;
            default:         rs = 3;      cs = 0;          break;
        }
        //  elements at offsets i*rs + j*cs, 0 <= i < r, 0 <= j < c
        std::int64_t lo = 0, hi = 0;
        if (r>0 && c>0) {
            lo = std::min<std::int64_t>(0, rs*(r-1))
               + std::min<std::int64_t>(0, cs*(c-1));
            hi = std::max<std::int64_t>(0, rs*(r-1))
               + std::max<std::int64_t>(0, cs*(c-1));
        }
        //  a margin on both sides, to detect writes outside the matrix
        base = 5 - lo;
        buf.resize(std::size_t(hi - lo + 11));
    }

    T *ptr()                              { return &buf[base]; }
    T &operator()(std::int64_t i, std::int64_t j)
                                          { return buf[base + i*rs + j*cs]; }

    std::int64_t    r, c, rs, cs, base;
    std::vector<T>  buf;
};

template <typename T>
void
randomize(Random &random, std::vector<T> &v)
{
    for (auto &x : v) {
        x = random.value<T>();
    }
}

template <typename T>
T
conjIf(bool c, const T &x)
{
    return c ? conjugate(x) : x;
}

//
//  One case: C = alpha op(A) op(B) + beta C, A m x k, B k x n, C m x n
//
template <typename T>
bool
stridedCase(Context &ctx, Random &random, std::int64_t m, std::int64_t n,
            std::int64_t k, Layout lA, Layout lB, Layout lC, bool conjA,
            bool conjB, T alpha, T beta)
{
    Operand<T> A(lA, m, k), B(lB, k, n), C(lC, m, n);

    randomize(random, A.buf);
    randomize(random, B.buf);
    randomize(random, C.buf);
    if (alpha==T(0)) {
        //  A and B must not be read
        for (auto &x : A.buf) x = nan<T>();
        for (auto &x : B.buf) x = nan<T>();
    }
    if (beta==T(0)) {
        //  C must not be read
        for (std::int64_t j=0; j<n; ++j) {
            for (std::int64_t i=0; i<m; ++i) {
                C(i,j) = nan<T>();
            }
        }
    }
    std::vector<T> C0 = C.buf;

    //  reference and error bound
    std::vector<cdouble> ref(std::size_t(std::max<std::int64_t>(m*n, 1)));
    std::vector<double>  bound(ref.size());
    for (std::int64_t j=0; j<n; ++j) {
        for (std::int64_t i=0; i<m; ++i) {
            cdouble s = 0;
            double  b = 0;
            if (alpha!=T(0)) {
                for (std::int64_t l=0; l<k; ++l) {
                    cdouble p = toCdouble(conjIf(conjA, A(i,l)))
                              * toCdouble(conjIf(conjB, B(l,j)));
                    s += p;
                    b += std::abs(p);
                }
            }
            cdouble a = toCdouble(alpha), be = toCdouble(beta);
            cdouble c = beta==T(0) ? cdouble(0) : toCdouble(C(i,j));
            ref[i+j*m] = a*s + be*c;
            bound[i+j*m] = std::abs(a)*b + std::abs(be)*std::abs(c);
        }
    }

    gemmStrided(m, n, k, alpha, conjA, A.ptr(), A.rs, A.cs, conjB, B.ptr(),
                B.rs, B.cs, beta, C.ptr(), C.rs, C.cs);

    std::string what = format("gemm_strided<%s>(m=%d,n=%d,k=%d,A %s%s,"
                              "B %s%s,C %s,alpha=%s,beta=%s)",
                              typeName<T>(), int(m), int(n), int(k),
                              layoutName(lA), conjA ? " conj" : "",
                              layoutName(lB), conjB ? " conj" : "",
                              layoutName(lC), str(alpha).c_str(),
                              str(beta).c_str());

    //  elements of C, and all other elements of its buffer unchanged
    std::vector<char> isC(C.buf.size(), 0);
    for (std::int64_t j=0; j<n; ++j) {
        for (std::int64_t i=0; i<m; ++i) {
            isC[std::size_t(C.base + i*C.rs + j*C.cs)] = 1;
            T t = fromCdouble<T>(ref[i+j*m]);
            if (!ctx.check(closeEnough(C(i,j), t, bound[i+j*m],
                                       int(std::max<std::int64_t>(k, 1))),
                           "%s: C(%d,%d) = %s, expected %s", what.c_str(),
                           int(i), int(j), str(C(i,j)).c_str(),
                           str(t).c_str()))
            {
                return false;
            }
        }
    }
    for (std::size_t p=0; p<C.buf.size(); ++p) {
        if (!isC[p] && !ctx.check(sameBits(C.buf[p], C0[p]),
                                  "%s: element %d of the buffer of C outside "
                                  "of C modified", what.c_str(), int(p)))
        {
            return false;
        }
    }
    return true;
}

template <typename T>
T
scalar(double re, double im)
{
    return fromCdouble<T>(cdouble(re, im));
}

//
//  All layouts of A, B and C (C without zero strides) for one size
//
template <typename T>
void
testLayouts(Context &ctx, std::int64_t m, std::int64_t n, std::int64_t k)
{
    Random random(0x5172ed + m*1000003 + n*1009 + k);
    const T alphas[] = { scalar<T>(1, 0), scalar<T>(-1.5, 0.5) };
    const T betas[]  = { scalar<T>(0, 0), scalar<T>(0.5, -0.25) };
    const bool cplx = IsComplex<T>::value;

    for (int lA=0; lA<NumLayouts; ++lA) {
        for (int lB=0; lB<NumLayouts; ++lB) {
            for (int lC=0; lC<ZeroRow; ++lC) {
                //  all combinations of A and B with column-major C; all
                //  layouts of C with A, B column major or negative
                if (lC!=ColMajor && !(lA==ColMajor && lB==ColMajor)
                    && !(lA==NegBoth && lB==Mixed))
                {
                    continue;
                }
                for (int conj=0; conj<(cplx ? 4 : 1); ++conj) {
                    T alpha = alphas[(lA+lB+conj) % 2];
                    T beta  = betas[(lA+lC+conj) % 2];
                    if (!stridedCase<T>(ctx, random, m, n, k, Layout(lA),
                                        Layout(lB), Layout(lC), conj & 1,
                                        conj & 2, alpha, beta))
                    {
                        return;
                    }
                }
            }
        }
    }
}

template <typename T>
void
testSizes(Context &ctx)
{
    //  tiny, partial tiles and blocks of the test configurations, and
    //  sizes beyond the default blocks
    static const int size[][3] = { {1,1,1}, {2,3,1}, {7,5,3}, {13,17,11},
                                   {33,31,65}, {64,64,64}, {97,113,301} };
    for (const auto &s : size) {
        testLayouts<T>(ctx, s[0], s[1], s[2]);
    }
}

template <typename T>
void
testLarge(Context &ctx)
{
    Random random(0x1a76e);
    const T alpha = scalar<T>(0.75, -0.5), beta = scalar<T>(-0.5, 0.25);
    const Layout layouts[][3] = { {ColMajor, ColMajor, ColMajor},
                                  {RowStride2, NegRow, RowMajor},
                                  {Mixed, NegBoth, RowStride2},
                                  {ZeroRow, RowMajor, NegRow} };
    for (const auto &l : layouts) {
        if (!stridedCase<T>(ctx, random, 413, 389, 1031, l[0], l[1], l[2],
                            IsComplex<T>::value, false, alpha, beta))
        {
            return;
        }
    }
}

//
//  alpha = 0, k = 0, empty C, and zero strides of C where they are allowed
//  (dimension of length 1)
//
template <typename T>
void
testSpecial(Context &ctx)
{
    Random random(0x5e1a1);
    const T one = scalar<T>(1, 0), zero = scalar<T>(0, 0);
    const T beta = scalar<T>(-2, 0.5);
    struct { std::int64_t m, n, k; T alpha, beta; } c[] = {
        {9, 7, 5, zero, beta}, {9, 7, 5, zero, zero}, {9, 7, 0, one, beta},
        {9, 7, 0, one, zero}, {0, 7, 5, one, beta}, {9, 0, 5, one, beta},
        {0, 0, 0, one, zero}
    };
    for (const auto &x : c) {
        if (!stridedCase<T>(ctx, random, x.m, x.n, x.k, RowStride2, NegRow,
                            Mixed, false, false, x.alpha, x.beta))
        {
            return;
        }
    }
    //  C a row vector with row stride 0, a column vector with column
    //  stride 0
    for (std::int64_t k : {1, 6, 300}) {
        if (!stridedCase<T>(ctx, random, 1, 37, k, Mixed, RowMajor, ZeroRow,
                            false, IsComplex<T>::value, one, beta)
         || !stridedCase<T>(ctx, random, 37, 1, k, NegRow, ColMajor, ZeroCol,
                            IsComplex<T>::value, false, one, zero))
        {
            return;
        }
    }
}

} // namespace

ULMBLAS_TEST(T0800, "strided/sgemm all layouts")   { testSizes<float>(ctx); }
ULMBLAS_TEST(T0801, "strided/dgemm all layouts")   { testSizes<double>(ctx); }
ULMBLAS_TEST(T0802, "strided/cgemm all layouts")   { testSizes<cfloat>(ctx); }
ULMBLAS_TEST(T0803, "strided/zgemm all layouts")   { testSizes<cdouble>(ctx); }
ULMBLAS_TEST(T0804, "strided/?gemm large") {
    testLarge<float>(ctx);
    testLarge<double>(ctx);
    testLarge<cfloat>(ctx);
    testLarge<cdouble>(ctx);
}
ULMBLAS_TEST(T0805, "strided/?gemm alpha=0, k=0, empty, zero strides of C") {
    testSpecial<float>(ctx);
    testSpecial<double>(ctx);
    testSpecial<cfloat>(ctx);
    testSpecial<cdouble>(ctx);
}
