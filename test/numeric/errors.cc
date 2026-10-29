//
//  Invalid arguments: a function of the C interface that detects an invalid
//  argument reports it through cblas_xerbla and returns without computing
//  (reference CBLAS: "cblas_xerbla(...); return;").  Regression tests for
//  issue #0005: the wrappers continued with the invalid arguments, which
//  accessed memory outside the operands.
//
//  This test program provides its own cblas_xerbla (as the Netlib CBLAS
//  test driver does), which records the reported parameter and routine and
//  returns.  Each case calls a function with dimension 4 and exactly one
//  invalid argument (a too small leading dimension or a zero increment).
//  The expected INFO is the position of that argument.  The operands point
//  into the middle of large buffers; the complete buffers must be unchanged
//  after the call (a function that continued would modify them).
//
#include BLAS_HEADER
#include <cstdarg>
#include <string>
#include <vector>
#include <test/numeric/testing.h>

namespace {

int          xerblaCalls;
int          xerblaInfo;
std::string  xerblaRoutine;

void
resetXerbla()
{
    xerblaCalls = 0;
    xerblaInfo  = 0;
    xerblaRoutine.clear();
}

//
//  Operands A, X, Y, B, C of real type T (complex operands use pairs of T)
//  and the scalars alpha and beta (real and complex).
//
template <typename T>
struct Buffers
{
    Buffers()
        : ral(2), rbe(3)
    {
        T *p[5];
        for (int k=0; k<5; ++k) {
            data[k].resize(size);
            for (int i=0; i<size; ++i) {
                data[k][i] = T(1) + T(k) + T(0.001)*T(i%997);
            }
            p[k] = &data[k][offset];
        }
        A = p[0]; X = p[1]; Y = p[2]; B = p[3]; C = p[4];
        al[0] = 2; al[1] = 1;
        be[0] = 3; be[1] = -1;
    }

    static const int  size = 4096, offset = 1024;

    std::vector<T>  data[5];
    T               *A, *X, *Y, *B, *C;
    T               ral, rbe;
    T               al[2], be[2];
};

template <typename T>
bool
unchanged(const Buffers<T> &b, const std::vector<T> (&before)[5])
{
    for (int k=0; k<5; ++k) {
        for (std::size_t i=0; i<before[k].size(); ++i) {
            if (!test::sameBits(b.data[k][i], before[k][i])) {
                return false;
            }
        }
    }
    return true;
}

bool
checkCase(test::Context &ctx, const char *name, int info, bool same)
{
    return ctx.check(xerblaCalls==1 && xerblaInfo==info
                     && xerblaRoutine==name && same,
                     "%s: xerbla called %d times, info %d (expected %d), "
                     "routine '%s'; operands %s", name, xerblaCalls,
                     xerblaInfo, info, xerblaRoutine.c_str(),
                     same ? "unchanged" : "MODIFIED (computation continued)");
}

#define ERRCASE(T, NAME, INFO, CALL)                                        \
    {                                                                       \
        Buffers<T>      b;                                                  \
        std::vector<T>  before[5];                                          \
        for (int k=0; k<5; ++k) {                                           \
            before[k] = b.data[k];                                          \
        }                                                                   \
        resetXerbla();                                                      \
        CALL;                                                               \
        checkCase(ctx, NAME, INFO, unchanged(b, before));                   \
    }

} // namespace

extern "C" void
cblas_xerbla(int info, const char *rout, const char *, ...)
{
    ++xerblaCalls;
    xerblaInfo    = info;
    xerblaRoutine = rout;
}

ULMBLAS_TEST(T0700, "errors/cblas_dsyr with n=4, ldA=1")
{
    ERRCASE(double, "cblas_dsyr", 8,
            cblas_dsyr(CblasColMajor, CblasLower, 4, b.ral, b.X, 1, b.A, 1));
}

ULMBLAS_TEST(T0701, "errors/invalid arguments of all checking CBLAS functions")
{
    // float
    ERRCASE(float, "cblas_sgemv",  7, cblas_sgemv(CblasColMajor, CblasNoTrans, 4, 4, b.ral, b.A, 1, b.X, 1, b.rbe, b.Y, 1));
    ERRCASE(float, "cblas_sgbmv",  9, cblas_sgbmv(CblasColMajor, CblasNoTrans, 4, 4, 1, 1, b.ral, b.A, 1, b.X, 1, b.rbe, b.Y, 1));
    ERRCASE(float, "cblas_ssymv",  6, cblas_ssymv(CblasColMajor, CblasUpper, 4, b.ral, b.A, 1, b.X, 1, b.rbe, b.Y, 1));
    ERRCASE(float, "cblas_ssbmv",  7, cblas_ssbmv(CblasColMajor, CblasUpper, 4, 1, b.ral, b.A, 1, b.X, 1, b.rbe, b.Y, 1));
    ERRCASE(float, "cblas_sspmv",  7, cblas_sspmv(CblasColMajor, CblasUpper, 4, b.ral, b.A, b.X, 0, b.rbe, b.Y, 1));
    ERRCASE(float, "cblas_sger", 10, cblas_sger(CblasColMajor, 4, 4, b.ral, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(float, "cblas_ssyr",  8, cblas_ssyr(CblasColMajor, CblasUpper, 4, b.ral, b.X, 1, b.A, 1));
    ERRCASE(float, "cblas_ssyr2", 10, cblas_ssyr2(CblasColMajor, CblasUpper, 4, b.ral, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(float, "cblas_sspr",  6, cblas_sspr(CblasColMajor, CblasUpper, 4, b.ral, b.X, 0, b.A));
    ERRCASE(float, "cblas_sspr2",  8, cblas_sspr2(CblasColMajor, CblasUpper, 4, b.ral, b.X, 1, b.Y, 0, b.A));
    ERRCASE(float, "cblas_ssymm",  8, cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, 4, 4, b.ral, b.A, 1, b.B, 4, b.rbe, b.C, 4));
    ERRCASE(float, "cblas_ssyrk",  8, cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.ral, b.A, 1, b.rbe, b.C, 4));
    ERRCASE(float, "cblas_ssyr2k",  8, cblas_ssyr2k(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.ral, b.A, 1, b.B, 4, b.rbe, b.C, 4));
    ERRCASE(float, "cblas_strmv",  7, cblas_strmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, 1, b.X, 1));
    ERRCASE(float, "cblas_stbmv",  8, cblas_stbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 1, b.A, 1, b.X, 1));
    ERRCASE(float, "cblas_stpmv",  8, cblas_stpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, b.X, 0));
    ERRCASE(float, "cblas_strsv",  7, cblas_strsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, 1, b.X, 1));
    ERRCASE(float, "cblas_stbsv",  8, cblas_stbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 1, b.A, 1, b.X, 1));
    ERRCASE(float, "cblas_stpsv",  8, cblas_stpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, b.X, 0));
    ERRCASE(float, "cblas_sgemm",  9, cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, b.ral, b.A, 1, b.B, 4, b.rbe, b.C, 4));
    ERRCASE(float, "cblas_strmm", 10, cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 4, b.ral, b.A, 1, b.B, 4));
    ERRCASE(float, "cblas_strsm", 10, cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 4, b.ral, b.A, 1, b.B, 4));
    // double
    ERRCASE(double, "cblas_dgemv",  7, cblas_dgemv(CblasColMajor, CblasNoTrans, 4, 4, b.ral, b.A, 1, b.X, 1, b.rbe, b.Y, 1));
    ERRCASE(double, "cblas_dgbmv",  9, cblas_dgbmv(CblasColMajor, CblasNoTrans, 4, 4, 1, 1, b.ral, b.A, 1, b.X, 1, b.rbe, b.Y, 1));
    ERRCASE(double, "cblas_dsymv",  6, cblas_dsymv(CblasColMajor, CblasUpper, 4, b.ral, b.A, 1, b.X, 1, b.rbe, b.Y, 1));
    ERRCASE(double, "cblas_dsbmv",  7, cblas_dsbmv(CblasColMajor, CblasUpper, 4, 1, b.ral, b.A, 1, b.X, 1, b.rbe, b.Y, 1));
    ERRCASE(double, "cblas_dspmv",  7, cblas_dspmv(CblasColMajor, CblasUpper, 4, b.ral, b.A, b.X, 0, b.rbe, b.Y, 1));
    ERRCASE(double, "cblas_dger", 10, cblas_dger(CblasColMajor, 4, 4, b.ral, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(double, "cblas_dsyr",  8, cblas_dsyr(CblasColMajor, CblasUpper, 4, b.ral, b.X, 1, b.A, 1));
    ERRCASE(double, "cblas_dsyr2", 10, cblas_dsyr2(CblasColMajor, CblasUpper, 4, b.ral, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(double, "cblas_dspr",  6, cblas_dspr(CblasColMajor, CblasUpper, 4, b.ral, b.X, 0, b.A));
    ERRCASE(double, "cblas_dspr2",  8, cblas_dspr2(CblasColMajor, CblasUpper, 4, b.ral, b.X, 1, b.Y, 0, b.A));
    ERRCASE(double, "cblas_dsymm",  8, cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, 4, 4, b.ral, b.A, 1, b.B, 4, b.rbe, b.C, 4));
    ERRCASE(double, "cblas_dsyrk",  8, cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.ral, b.A, 1, b.rbe, b.C, 4));
    ERRCASE(double, "cblas_dsyr2k",  8, cblas_dsyr2k(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.ral, b.A, 1, b.B, 4, b.rbe, b.C, 4));
    ERRCASE(double, "cblas_dtrmv",  7, cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, 1, b.X, 1));
    ERRCASE(double, "cblas_dtbmv",  8, cblas_dtbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 1, b.A, 1, b.X, 1));
    ERRCASE(double, "cblas_dtpmv",  8, cblas_dtpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, b.X, 0));
    ERRCASE(double, "cblas_dtrsv",  7, cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, 1, b.X, 1));
    ERRCASE(double, "cblas_dtbsv",  8, cblas_dtbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 1, b.A, 1, b.X, 1));
    ERRCASE(double, "cblas_dtpsv",  8, cblas_dtpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, b.X, 0));
    ERRCASE(double, "cblas_dgemm",  9, cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, b.ral, b.A, 1, b.B, 4, b.rbe, b.C, 4));
    ERRCASE(double, "cblas_dtrmm", 10, cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 4, b.ral, b.A, 1, b.B, 4));
    ERRCASE(double, "cblas_dtrsm", 10, cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 4, b.ral, b.A, 1, b.B, 4));
    // float (complex)
    ERRCASE(float, "cblas_cgemv",  7, cblas_cgemv(CblasColMajor, CblasNoTrans, 4, 4, b.al, b.A, 1, b.X, 1, b.be, b.Y, 1));
    ERRCASE(float, "cblas_cgbmv",  9, cblas_cgbmv(CblasColMajor, CblasNoTrans, 4, 4, 1, 1, b.al, b.A, 1, b.X, 1, b.be, b.Y, 1));
    ERRCASE(float, "cblas_chemv",  6, cblas_chemv(CblasColMajor, CblasUpper, 4, b.al, b.A, 1, b.X, 1, b.be, b.Y, 1));
    ERRCASE(float, "cblas_chbmv",  7, cblas_chbmv(CblasColMajor, CblasUpper, 4, 1, b.al, b.A, 1, b.X, 1, b.be, b.Y, 1));
    ERRCASE(float, "cblas_chpmv",  7, cblas_chpmv(CblasColMajor, CblasUpper, 4, b.al, b.A, b.X, 0, b.be, b.Y, 1));
    ERRCASE(float, "cblas_cgeru", 10, cblas_cgeru(CblasColMajor, 4, 4, b.al, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(float, "cblas_cgerc", 10, cblas_cgerc(CblasColMajor, 4, 4, b.al, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(float, "cblas_cher",  8, cblas_cher(CblasColMajor, CblasUpper, 4, b.ral, b.X, 1, b.A, 1));
    ERRCASE(float, "cblas_cher2", 10, cblas_cher2(CblasColMajor, CblasUpper, 4, b.al, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(float, "cblas_chpr",  6, cblas_chpr(CblasColMajor, CblasUpper, 4, b.ral, b.X, 0, b.A));
    ERRCASE(float, "cblas_chpr2",  8, cblas_chpr2(CblasColMajor, CblasUpper, 4, b.al, b.X, 1, b.Y, 0, b.A));
    ERRCASE(float, "cblas_csymm",  8, cblas_csymm(CblasColMajor, CblasLeft, CblasUpper, 4, 4, b.al, b.A, 1, b.B, 4, b.be, b.C, 4));
    ERRCASE(float, "cblas_chemm",  8, cblas_chemm(CblasColMajor, CblasLeft, CblasUpper, 4, 4, b.al, b.A, 1, b.B, 4, b.be, b.C, 4));
    ERRCASE(float, "cblas_csyrk",  8, cblas_csyrk(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.al, b.A, 1, b.be, b.C, 4));
    ERRCASE(float, "cblas_cherk",  8, cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.ral, b.A, 1, b.rbe, b.C, 4));
    ERRCASE(float, "cblas_csyr2k",  8, cblas_csyr2k(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.al, b.A, 1, b.B, 4, b.be, b.C, 4));
    ERRCASE(float, "cblas_cher2k",  8, cblas_cher2k(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.al, b.A, 1, b.B, 4, b.rbe, b.C, 4));
    ERRCASE(float, "cblas_ctrmv",  7, cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, 1, b.X, 1));
    ERRCASE(float, "cblas_ctbmv",  8, cblas_ctbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 1, b.A, 1, b.X, 1));
    ERRCASE(float, "cblas_ctpmv",  8, cblas_ctpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, b.X, 0));
    ERRCASE(float, "cblas_ctrsv",  7, cblas_ctrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, 1, b.X, 1));
    ERRCASE(float, "cblas_ctbsv",  8, cblas_ctbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 1, b.A, 1, b.X, 1));
    ERRCASE(float, "cblas_ctpsv",  8, cblas_ctpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, b.X, 0));
    ERRCASE(float, "cblas_cgemm",  9, cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, b.al, b.A, 1, b.B, 4, b.be, b.C, 4));
    ERRCASE(float, "cblas_ctrmm", 10, cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 4, b.al, b.A, 1, b.B, 4));
    ERRCASE(float, "cblas_ctrsm", 10, cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 4, b.al, b.A, 1, b.B, 4));
    // double (complex)
    ERRCASE(double, "cblas_zgemv",  7, cblas_zgemv(CblasColMajor, CblasNoTrans, 4, 4, b.al, b.A, 1, b.X, 1, b.be, b.Y, 1));
    ERRCASE(double, "cblas_zgbmv",  9, cblas_zgbmv(CblasColMajor, CblasNoTrans, 4, 4, 1, 1, b.al, b.A, 1, b.X, 1, b.be, b.Y, 1));
    ERRCASE(double, "cblas_zhemv",  6, cblas_zhemv(CblasColMajor, CblasUpper, 4, b.al, b.A, 1, b.X, 1, b.be, b.Y, 1));
    ERRCASE(double, "cblas_zhbmv",  7, cblas_zhbmv(CblasColMajor, CblasUpper, 4, 1, b.al, b.A, 1, b.X, 1, b.be, b.Y, 1));
    ERRCASE(double, "cblas_zhpmv",  7, cblas_zhpmv(CblasColMajor, CblasUpper, 4, b.al, b.A, b.X, 0, b.be, b.Y, 1));
    ERRCASE(double, "cblas_zgeru", 10, cblas_zgeru(CblasColMajor, 4, 4, b.al, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(double, "cblas_zgerc", 10, cblas_zgerc(CblasColMajor, 4, 4, b.al, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(double, "cblas_zher",  8, cblas_zher(CblasColMajor, CblasUpper, 4, b.ral, b.X, 1, b.A, 1));
    ERRCASE(double, "cblas_zher2", 10, cblas_zher2(CblasColMajor, CblasUpper, 4, b.al, b.X, 1, b.Y, 1, b.A, 1));
    ERRCASE(double, "cblas_zhpr",  6, cblas_zhpr(CblasColMajor, CblasUpper, 4, b.ral, b.X, 0, b.A));
    ERRCASE(double, "cblas_zhpr2",  8, cblas_zhpr2(CblasColMajor, CblasUpper, 4, b.al, b.X, 1, b.Y, 0, b.A));
    ERRCASE(double, "cblas_zsymm",  8, cblas_zsymm(CblasColMajor, CblasLeft, CblasUpper, 4, 4, b.al, b.A, 1, b.B, 4, b.be, b.C, 4));
    ERRCASE(double, "cblas_zhemm",  8, cblas_zhemm(CblasColMajor, CblasLeft, CblasUpper, 4, 4, b.al, b.A, 1, b.B, 4, b.be, b.C, 4));
    ERRCASE(double, "cblas_zsyrk",  8, cblas_zsyrk(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.al, b.A, 1, b.be, b.C, 4));
    ERRCASE(double, "cblas_zherk",  8, cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.ral, b.A, 1, b.rbe, b.C, 4));
    ERRCASE(double, "cblas_zsyr2k",  8, cblas_zsyr2k(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.al, b.A, 1, b.B, 4, b.be, b.C, 4));
    ERRCASE(double, "cblas_zher2k",  8, cblas_zher2k(CblasColMajor, CblasUpper, CblasNoTrans, 4, 4, b.al, b.A, 1, b.B, 4, b.rbe, b.C, 4));
    ERRCASE(double, "cblas_ztrmv",  7, cblas_ztrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, 1, b.X, 1));
    ERRCASE(double, "cblas_ztbmv",  8, cblas_ztbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 1, b.A, 1, b.X, 1));
    ERRCASE(double, "cblas_ztpmv",  8, cblas_ztpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, b.X, 0));
    ERRCASE(double, "cblas_ztrsv",  7, cblas_ztrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, 1, b.X, 1));
    ERRCASE(double, "cblas_ztbsv",  8, cblas_ztbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 1, b.A, 1, b.X, 1));
    ERRCASE(double, "cblas_ztpsv",  8, cblas_ztpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4, b.A, b.X, 0));
    ERRCASE(double, "cblas_zgemm",  9, cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, b.al, b.A, 1, b.B, 4, b.be, b.C, 4));
    ERRCASE(double, "cblas_ztrmm", 10, cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 4, b.al, b.A, 1, b.B, 4));
    ERRCASE(double, "cblas_ztrsm", 10, cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 4, 4, b.al, b.A, 1, b.B, 4));
}
