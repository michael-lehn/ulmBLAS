//
//  Indices beyond INT_MAX (T0356, issue #0015): operands whose leading
//  dimension or increment fits into an int, while the offset of the last
//  element (e.g. 2*ldC) does not.  The operands live in a region of 8 to
//  16 GiB of address space reserved with mmap(MAP_NORESERVE); only the
//  pages of the few elements that are used are touched.  With a 64-bit
//  ULMBLAS_INDEX_TYPE also a leading dimension beyond INT_MAX is used.
//
#include BLAS_HEADER
#include <test/numeric/testing.h>

#include <climits>
#include <cstddef>
#include <cstdio>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#define ULMBLAS_TEST_HAVE_MMAP 1
#endif

namespace {

using namespace test;

typedef ULMBLAS_INDEX_TYPE  Index;
typedef std::ptrdiff_t      Offset;

//  2*L > INT_MAX, L < INT_MAX
const Offset L = (Offset(1) << 30) + 7;

//
//  A region of n floats, zero initialized, or a null pointer if the
//  address space cannot be reserved (32-bit targets, limits).
//
class Region
{
    public:
        explicit
        Region(Offset n)
            : p(0), bytes(std::size_t(n)*sizeof(float))
        {
#if defined(ULMBLAS_TEST_HAVE_MMAP)
            if (sizeof(void *) >= 8) {
                void *q = mmap(0, bytes, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
                if (q != MAP_FAILED) {
                    p = static_cast<float *>(q);
                }
            }
#endif
        }

        ~Region()
        {
#if defined(ULMBLAS_TEST_HAVE_MMAP)
            if (p) {
                munmap(p, bytes);
            }
#endif
        }

        float        *p;
        std::size_t  bytes;

    private:
        Region(const Region &);
        Region &operator=(const Region &);
};

bool
reserved(Context &ctx, const Region &r)
{
    if (!r.p) {
        std::fprintf(stderr, "note: T0356: cannot reserve %zu bytes of "
                     "address space, case not run\n", r.bytes);
        ctx.pass();
        return false;
    }
    return true;
}

bool
near(float x, float y)
{
    return x - y < 1e-5f && y - x < 1e-5f;
}

//
//  C(i,j) at C[i*rsC + j*csC]: C <- A*B + C for A = [1 2; 3 4] (2 x 2), B =
//  [1 2 3; 4 5 6] (2 x 3) and C = 1, i.e. C = [10 13 16; 20 27 34].
//
bool
checkC(Context &ctx, const char *what, const float *C, Offset rsC,
       Offset csC)
{
    const float expected[2][3] = { { 10, 13, 16 }, { 20, 27, 34 } };
    for (int i=0; i<2; ++i) {
        for (int j=0; j<3; ++j) {
            float c = C[i*rsC + j*csC];
            if (!ctx.check(near(c, expected[i][j]), "%s: C(%d,%d) = %g, "
                           "expected %g", what, i, j, c, expected[i][j]))
            {
                return false;
            }
        }
    }
    return true;
}

//  sgemm, column major, C with leading dimension ld (2 x 3)
bool
gemmC(Context &ctx, Offset ld)
{
    Region r(2*ld + 2);
    if (!reserved(ctx, r)) {
        return true;
    }
    const float A[] = { 1, 3, 2, 4 };
    const float B[] = { 1, 4, 2, 5, 3, 6 };
    for (int j=0; j<3; ++j) {
        r.p[j*ld] = r.p[1 + j*ld] = 1;
    }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 3, 2,
                1.0f, A, 2, B, 2, 1.0f, r.p, Index(ld));
    return checkC(ctx, "sgemm, large ldC", r.p, 1, ld);
}

//  sgemm, column major, both transposed: C (3 x 2) <- X^T Y^T + C with X
//  = B (2 x 3, leading dimension L) and Y = A of checkC, i.e. the
//  transposed product (A B)^T + C
bool
gemmATrans(Context &ctx)
{
    Region r(2*L + 2);
    if (!reserved(ctx, r)) {
        return true;
    }
    const float B[2][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
    const float A[]     = { 1, 3, 2, 4 };
    float       C[6]    = { 1, 1, 1, 1, 1, 1 };
    for (int l=0; l<2; ++l) {
        for (int i=0; i<3; ++i) {
            r.p[l + i*L] = B[l][i];
        }
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, 3, 2, 2,
                1.0f, r.p, Index(L), A, 2, 1.0f, C, 3);
    return checkC(ctx, "sgemm, transposed A with large ldA", C, 3, 1);
}

//  sgemm, row major: C^T (3 x 2, rows L apart) <- B^T A^T + C^T with A, B
//  of checkC
bool
gemmRowMajor(Context &ctx)
{
    Region s(2*L + 2);
    if (!reserved(ctx, s)) {
        return true;
    }
    const float Bt[] = { 1, 4, 2, 5, 3, 6 };      // 3 x 2, row major
    const float At[] = { 1, 3, 2, 4 };            // 2 x 2, row major
    for (int i=0; i<3; ++i) {
        s.p[i*L] = s.p[1 + i*L] = 1;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 2, 2,
                1.0f, Bt, 2, At, 2, 1.0f, s.p, Index(L));
    return checkC(ctx, "sgemm, row major, large ldC", s.p, 1, L);
}

//  strsm: B (2 x 3, ld L) <- inv(A) B with A = [1 0; 3 4] lower triangular
bool
trsm(Context &ctx)
{
    Region r(2*L + 2);
    if (!reserved(ctx, r)) {
        return true;
    }
    const float A[] = { 1, 3, 0, 4 };
    const float X[2][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
    for (int j=0; j<3; ++j) {
        r.p[j*L]     = X[0][j];
        r.p[1 + j*L] = 3*X[0][j] + 4*X[1][j];
    }
    cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, 2, 3, 1.0f, A, 2, r.p, Index(L));
    for (int i=0; i<2; ++i) {
        for (int j=0; j<3; ++j) {
            float x = r.p[i + j*L];
            if (!ctx.check(near(x, X[i][j]), "strsm, large ldB: X(%d,%d) "
                           "= %g, expected %g", i, j, x, X[i][j]))
            {
                return false;
            }
        }
    }
    return true;
}

//  sgemv: y (3 elements, increment L) <- A x + y, A (3 x 2, ld L)
bool
gemv(Context &ctx)
{
    Region a(L + 3), y(2*L + 1);
    if (!reserved(ctx, a) || !reserved(ctx, y)) {
        return true;
    }
    const float x[] = { 1, 2 };
    for (int i=0; i<3; ++i) {
        a.p[i]     = float(i + 1);              // column 0: 1 2 3
        a.p[i + L] = float(i + 4);              // column 1: 4 5 6
        y.p[i*L]   = 1;
    }
    cblas_sgemv(CblasColMajor, CblasNoTrans, 3, 2, 1.0f, a.p, Index(L),
                x, 1, 1.0f, y.p, Index(L));
    const float expected[] = { 10, 13, 16 };
    for (int i=0; i<3; ++i) {
        if (!ctx.check(near(y.p[i*L], expected[i]), "sgemv, large ldA and "
                       "incY: y(%d) = %g, expected %g", i, y.p[i*L],
                       expected[i]))
        {
            return false;
        }
    }
    return true;
}

//  saxpy: y (3 elements, increment L) <- 2 x + y
bool
axpy(Context &ctx)
{
    Region y(2*L + 1);
    if (!reserved(ctx, y)) {
        return true;
    }
    const float x[] = { 1, 2, 3 };
    for (int i=0; i<3; ++i) {
        y.p[i*L] = 1;
    }
    cblas_saxpy(3, 2.0f, x, 1, y.p, Index(L));
    for (int i=0; i<3; ++i) {
        if (!ctx.check(near(y.p[i*L], 2*x[i] + 1), "saxpy, large incY: "
                       "y(%d) = %g", i, y.p[i*L]))
        {
            return false;
        }
    }
    return true;
}

} // namespace

ULMBLAS_TEST(T0356, "interface/indices: offsets of elements beyond INT_MAX "
             "(sgemm, strsm, sgemv, saxpy; leading dimensions and "
             "increments of about 2^30, and beyond INT_MAX with a 64-bit "
             "ULMBLAS_INDEX_TYPE)")
{
    bool ok = gemmC(ctx, L) && gemmATrans(ctx) && gemmRowMajor(ctx)
           && trsm(ctx) && gemv(ctx) && axpy(ctx);
    if (ok && sizeof(Index) >= 8) {
        ok = gemmC(ctx, Offset(INT_MAX) + 4);
    }
    if (ok) {
        ctx.pass();
    }
}
