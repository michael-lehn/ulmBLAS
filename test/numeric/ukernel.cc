//
//  Tests of the GEMM micro kernel (internal C++ interface of ulmBLAS).
//
//  The micro kernel selected by the configuration (ulmBLAS::ugemm, e.g. the
//  SSE kernel for double on x86-64) and the reference micro kernel
//  (ulmBLAS::ref::ugemm) are both compared with a straightforward
//  computation of
//
//      C <- beta*C + alpha*A*B,    A: MR x kc packed,  B: kc x NR packed
//
//  using error bounds (see test::closeEnough).  If beta is zero, C contains
//  NaN and Inf on entry: BLAS does not require C to be set in this case.
//  Packed operands are allocated by the memory pool used by gemm (aligned as
//  the selected kernels require).  C is tested with column major, row major
//  and general strides; elements outside the tile must not be modified.
//
//  Partial tiles (mr < MR or nr < NR) are computed by the buffered variant
//  ulmBLAS::ugemm(mr, nr, kc, ...), with operands packed by gepack_A and
//  gepack_B (zero padded), as in the macro kernel.
//
//  Note: where no optimized kernel exists for a type (and on platforms
//  without an optimized kernel) ugemm is ref::ugemm.
//
#include <ulmblas/ulmblas.h>
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

namespace {

using namespace test;

const int kcs[]  = { 0, 1, 2, 3, 4, 5, 7, 8, 13, 64, 385 };
const int numKc  = sizeof(kcs)/sizeof(kcs[0]);

// strides of C: column major, row major (both with padding), general
enum Layout { ColMajor, RowMajor, General };

struct Strides
{
    Strides(Layout layout, int m, int n)
        : incRow(layout==ColMajor ? 1 : layout==RowMajor ? n+2 : 2),
          incCol(layout==ColMajor ? m+3 : layout==RowMajor ? 1 : 2*m+5),
          size((m-1)*incRow + (n-1)*incCol + 1 + 7)
    {
    }

    int incRow, incCol, size;
};

const char *
layoutName(Layout layout)
{
    return layout==ColMajor ? "col" : layout==RowMajor ? "row" : "general";
}

//
//  Values of C on entry if beta is zero
//
template <typename T>
T
special(int k)
{
    typedef typename Real<T>::type R;
    const R inf = std::numeric_limits<R>::infinity();
    return k%3==0 ? nan<T>() : k%3==1 ? T(inf) : T(-inf);
}

//
//  C buffer with an m x n tile and guard elements
//
template <typename T>
std::vector<T>
makeC(Random &random, const Strides &s, int m, int n, bool beta0)
{
    std::vector<T> C(s.size);
    for (std::size_t p=0; p<C.size(); ++p) {
        C[p] = random.value<T>();
    }
    if (beta0) {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                C[i*s.incRow+j*s.incCol] = special<T>(i+j*m);
            }
        }
    }
    return C;
}

//
//  Check the m x n tile of C against beta*C0 + alpha*A*B (A, B packed with
//  MR and NR), and that no element outside the tile was modified.
//
template <typename T>
bool
checkTile(Context &ctx, const std::string &what, int m, int n, int kc,
          int MR, int NR, const T &alpha, const T *A, const T *B,
          const T &beta, const std::vector<T> &C0, const std::vector<T> &C,
          const Strides &s)
{
    for (int j=0; j<n; ++j) {
        for (int i=0; i<m; ++i) {
            cdouble sum(0);
            double  bound = 0;
            for (int l=0; l<kc; ++l) {
                cdouble a = toCdouble(A[l*MR+i]), b = toCdouble(B[l*NR+j]);
                sum   += a*b;
                bound += std::abs(a)*std::abs(b);
            }
            sum   *= toCdouble(alpha);
            bound *= absValue(alpha);
            const T &c0 = C0[i*s.incRow+j*s.incCol];
            if (beta!=T(0)) {
                sum   += toCdouble(beta)*toCdouble(c0);
                bound += absValue(beta)*absValue(c0);
            }
            const T &c = C[i*s.incRow+j*s.incCol];
            if (!closeEnough(c, fromCdouble<T>(sum), bound, kc)) {
                return ctx.check(false, "%s: C(%d,%d) is %s, expected %s "
                                 "(C on entry %s), %s", what.c_str(), i, j,
                                 str(c).c_str(),
                                 str(fromCdouble<T>(sum)).c_str(),
                                 str(c0).c_str(), typeName<T>());
            }
        }
    }
    std::vector<bool> inTile(C.size(), false);
    for (int j=0; j<n; ++j) {
        for (int i=0; i<m; ++i) {
            inTile[i*s.incRow+j*s.incCol] = true;
        }
    }
    for (std::size_t p=0; p<C.size(); ++p) {
        if (!inTile[p] && !sameBits(C[p], C0[p])) {
            return ctx.check(false, "%s: element %d outside the tile "
                             "modified, %s", what.c_str(), int(p),
                             typeName<T>());
        }
    }
    ctx.pass();
    return true;
}

//
//  Full MR x NR tiles: selected kernel and reference kernel
//
template <typename T>
bool
testFullTiles(Context &ctx)
{
    const int MR = ulmBLAS::BlockSizeUGemm<T>::MR;
    const int NR = ulmBLAS::BlockSizeUGemm<T>::NR;

    static ulmBLAS::MemoryPool<T> pool;
    Random                        random;
    std::vector<T>                s = scalars<T>();

    for (int k=0; k<numKc; ++k) {
        int kc = kcs[k];
        T   *A = pool.allocate(MR*kc+MR);
        T   *B = pool.allocate(NR*kc+NR);
        for (int p=0; p<MR*kc; ++p) A[p] = random.value<T>();
        for (int p=0; p<NR*kc; ++p) B[p] = random.value<T>();

#if defined(USE_SSE)
        if (!ctx.check(ulmBLAS::isAligned(A, 16) && ulmBLAS::isAligned(B, 16),
                       "packed buffers not aligned to 16 bytes"))
        {
            return false;
        }
#endif
        bool ok = true;
        for (int layout=0; layout<3 && ok; ++layout) {
            Strides st(Layout(layout), MR, NR);
            for (std::size_t a=0; a<s.size() && ok; ++a) {
                for (std::size_t b=0; b<s.size() && ok; ++b) {
                    std::vector<T> C0 = makeC<T>(random, st, MR, NR,
                                                 s[b]==T(0));
                    std::vector<T> C1(C0), C2(C0);

                    ulmBLAS::ref::ugemm(kc, s[a], A, B, s[b], &C1[0],
                                        st.incRow, st.incCol, A, B);
                    ulmBLAS::ugemm(kc, s[a], A, B, s[b], &C2[0],
                                   st.incRow, st.incCol, A, B);

                    std::string what = format("ugemm(kc=%d,alpha=%s,beta=%s,"
                                              "C %s)", kc, str(s[a]).c_str(),
                                              str(s[b]).c_str(),
                                              layoutName(Layout(layout)));
                    ok = checkTile(ctx, "ref::" + what, MR, NR, kc, MR, NR,
                                   s[a], A, B, s[b], C0, C1, st)
                      && checkTile(ctx, "selected " + what, MR, NR, kc, MR,
                                   NR, s[a], A, B, s[b], C0, C2, st);
                }
            }
        }
        pool.release(A);
        pool.release(B);
        if (!ok) {
            return false;
        }
    }
    return true;
}

//
//  Partial tiles: buffered variant ugemm(mr, nr, kc, ...) with operands
//  packed (and zero padded) by gepack_A and gepack_B.  With specialValues
//  only beta = 0 with NaN and Inf in C is tested, otherwise all alpha and
//  beta with finite C.
//
template <typename T>
bool
testPartialTiles(Context &ctx, bool generalStrides, bool specialValues)
{
    const int MR = ulmBLAS::BlockSizeUGemm<T>::MR;
    const int NR = ulmBLAS::BlockSizeUGemm<T>::NR;
    const int kcList[] = { 0, 1, 5, 13 };

    static ulmBLAS::MemoryPool<T> pool;
    Random                        random(3);
    std::vector<T>                s = scalars<T>();

    for (int k=0; k<4; ++k) {
        int kc = kcList[k];
        T   *A_ = pool.allocate(MR*kc+MR);
        T   *B_ = pool.allocate(NR*kc+NR);
        bool ok = true;

        for (int mr=1; mr<=MR && ok; ++mr) {
            for (int nr=1; nr<=NR && ok; ++nr) {
                // mr x kc (column major) and kc x nr (row major) operands
                std::vector<T> A(std::max(1, mr*kc)), B(std::max(1, kc*nr));
                for (std::size_t p=0; p<A.size(); ++p) A[p] = random.value<T>();
                for (std::size_t p=0; p<B.size(); ++p) B[p] = random.value<T>();
                ulmBLAS::gepack_A(mr, kc, false, &A[0], 1, mr, A_);
                ulmBLAS::gepack_B(kc, nr, false, &B[0], nr, 1, B_);

                Layout layouts[2] = { ColMajor, RowMajor };
                if (generalStrides) {
                    layouts[0] = layouts[1] = General;
                }
                for (int l=0; l<(generalStrides ? 1 : 2) && ok; ++l) {
                    Strides st(layouts[l], mr, nr);
                    for (std::size_t a=0; a<s.size() && ok; ++a) {
                        for (std::size_t b=0; b<s.size() && ok; ++b) {
                            std::vector<T> C0 = makeC<T>(random, st, mr, nr,
                                                         s[b]==T(0)
                                                         && specialValues);
                            if (specialValues && s[b]!=T(0)) {
                                continue;
                            }
                            std::vector<T> C(C0);
                            ulmBLAS::ugemm(mr, nr, kc, s[a], A_, B_, s[b],
                                           &C[0], st.incRow, st.incCol,
                                           A_, B_);
                            std::string what = format(
                                "ugemm(mr=%d,nr=%d,kc=%d,alpha=%s,beta=%s,"
                                "C %s)", mr, nr, kc, str(s[a]).c_str(),
                                str(s[b]).c_str(), layoutName(layouts[l]));
                            ok = checkTile(ctx, what, mr, nr, kc, MR, NR,
                                           s[a], A_, B_, s[b], C0, C, st);
                        }
                    }
                }
            }
        }
        pool.release(A_);
        pool.release(B_);
        if (!ok) {
            return false;
        }
    }
    return true;
}

} // namespace

ULMBLAS_TEST(T0340, "ukernel/ugemm full tiles")
{
    testFullTiles<float>(ctx) && testFullTiles<double>(ctx)
    && testFullTiles<cfloat>(ctx) && testFullTiles<cdouble>(ctx);
}

ULMBLAS_TEST(T0341, "ukernel/ugemm partial tiles, unit stride C")
{
    testPartialTiles<float>(ctx, false, true)
    && testPartialTiles<float>(ctx, false, false)
    && testPartialTiles<double>(ctx, false, true)
    && testPartialTiles<double>(ctx, false, false)
    && testPartialTiles<cfloat>(ctx, false, true)
    && testPartialTiles<cfloat>(ctx, false, false)
    && testPartialTiles<cdouble>(ctx, false, true)
    && testPartialTiles<cdouble>(ctx, false, false);
}

ULMBLAS_TEST(T0342, "ukernel/ugemm partial tiles, general strides of C")
{
    testPartialTiles<float>(ctx, true, false)
    && testPartialTiles<double>(ctx, true, false)
    && testPartialTiles<cfloat>(ctx, true, false)
    && testPartialTiles<cdouble>(ctx, true, false);
}

ULMBLAS_TEST_XFAIL(T0343, "ukernel/ugemm partial tiles, general strides, "
                   "beta=0 with NaN and Inf in C", 6)
{
    testPartialTiles<float>(ctx, true, true)
    && testPartialTiles<double>(ctx, true, true)
    && testPartialTiles<cfloat>(ctx, true, true)
    && testPartialTiles<cdouble>(ctx, true, true);
}
