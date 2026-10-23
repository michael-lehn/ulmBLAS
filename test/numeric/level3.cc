//
//  Numerical tests of the level 3 functions of the C interface.
//
#include <test/numeric/blas.h>
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

namespace {

using namespace test;

const CBLAS_TRANSPOSE  transposes[] = { CblasNoTrans, CblasTrans,
                                        CblasConjTrans };
const CBLAS_UPLO       upLos[]      = { CblasLower, CblasUpper };
const CBLAS_SIDE       sides[]      = { CblasLeft, CblasRight };
const CBLAS_DIAG       diags[]      = { CblasNonUnit, CblasUnit };

struct Dims { int m, n, k; };

//
//  Small and odd dimensions, including zero dimensions.  The test
//  configurations with small block sizes (test/config) make these cover
//  partial and multiple blocks.
//
const Dims dims[] = { {0,3,2}, {3,0,2}, {3,2,0}, {1,1,1}, {2,3,1},
                      {5,7,3}, {17,13,19}, {33,31,40} };

const int numDims = sizeof(dims)/sizeof(dims[0]);

template <typename T>
void
fillNan(Matrix<T> &A)
{
    for (int j=0; j<A.n; ++j) {
        for (int i=0; i<A.m; ++i) {
            A(i,j) = nan<T>();
        }
    }
}

//
//  GEMM: C = alpha*op(A)*op(B) + beta*C
//
template <typename T>
bool
gemmCase(Context &ctx, Random &random, bool rowMajor, CBLAS_TRANSPOSE transA,
         CBLAS_TRANSPOSE transB, int m, int n, int k, T alpha, T beta)
{
    bool       tA = transA!=CblasNoTrans, tB = transB!=CblasNoTrans;
    Matrix<T>  A(tA ? k : m, tA ? m : k, rowMajor);
    Matrix<T>  B(tB ? n : k, tB ? k : n, rowMajor);
    Matrix<T>  C(m, n, rowMajor);

    A.randomize(random);
    B.randomize(random);
    C.randomize(random);
    if (beta==T(0)) {
        fillNan(C);
    }

    Result     ref = multiply(toCdouble(alpha), full(A, transA),
                              full(B, transB), toCdouble(beta), full(C));
    Matrix<T>  C0 = C;

    blas::gemm(order(rowMajor), transA, transB, m, n, k,
               alpha, A.ptr(), A.ld, B.ptr(), B.ld, beta, C.ptr(), C.ld);

    std::string what = format("gemm(%s,%c,%c,m=%d,n=%d,k=%d,alpha=%s,"
                              "beta=%s)", orderName(order(rowMajor)),
                              transChar(transA), transChar(transB), m, n, k,
                              str(alpha).c_str(), str(beta).c_str());
    return checkMatrix(ctx, C, ref, k, 'A', what)
        && ctx.check(C.paddingUnchanged(C0), "%s: padding of C modified",
                     what.c_str());
}

template <typename T>
void
testGemm(Context &ctx)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int o=0; o<2; ++o) {
        for (int ta=0; ta<3; ++ta) {
            for (int tb=0; tb<3; ++tb) {
                for (int d=0; d<numDims; ++d) {
                    for (std::size_t a=0; a<s.size(); ++a) {
                        for (std::size_t b=0; b<s.size(); ++b) {
                            if (!gemmCase(ctx, random, o==1, transposes[ta],
                                          transposes[tb], dims[d].m,
                                          dims[d].n, dims[d].k, s[a], s[b]))
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}

//
//  GEMM with dimensions larger than the default block sizes MC and KC of all
//  types (see ulmblas/config/blocksize.h), so that the production
//  configuration is tested with multiple and partial blocks.
//
template <typename T>
void
testGemmLarge(Context &ctx)
{
    Random random(7);
    T      alpha = fromCdouble<T>(cdouble(0.5, -0.25));
    T      beta  = fromCdouble<T>(cdouble(-1.5, 0.75));

    gemmCase(ctx, random, false, CblasNoTrans, CblasNoTrans,
             801, 37, 403, alpha, beta)
    && gemmCase(ctx, random, false, CblasConjTrans, CblasTrans,
                389, 21, 790, alpha, beta)
    && gemmCase(ctx, random, true, CblasNoTrans, CblasConjTrans,
                405, 19, 397, alpha, beta)
    && gemmCase(ctx, random, true, CblasTrans, CblasNoTrans,
                30, 401, 385, alpha, beta);
}

//
//  SYMM, HEMM: C = alpha*A*B + beta*C (side left) or C = alpha*B*A + beta*C
//  (side right) with A symmetric/hermitian, only triangle upLo referenced.
//
template <typename T>
bool
symmCase(Context &ctx, Random &random, bool hermitian, bool rowMajor,
         CBLAS_SIDE side, CBLAS_UPLO upLo, int m, int n, T alpha, T beta)
{
    int        na = (side==CblasLeft) ? m : n;
    Matrix<T>  A(na, na, rowMajor), B(m, n, rowMajor), C(m, n, rowMajor);

    A.randomize(random);
    B.randomize(random);
    C.randomize(random);
    poisonTriangle(A, upLo, false);
    if (beta==T(0)) {
        fillNan(C);
    }
    if (hermitian) {
        // imaginary parts of the diagonal are not referenced
        for (int i=0; i<na && alpha!=T(0); ++i) {
            A(i,i) = fromCdouble<T>(cdouble(std::real(A(i,i)), 77));
        }
    }

    Full    fA  = fullSymmetric(A, upLo, hermitian);
    Result  ref = (side==CblasLeft)
                ? multiply(toCdouble(alpha), fA, full(B), toCdouble(beta),
                           full(C))
                : multiply(toCdouble(alpha), full(B), fA, toCdouble(beta),
                           full(C));
    Matrix<T>  C0 = C;

    if (hermitian) {
        blas::hemm(order(rowMajor), side, upLo, m, n, alpha, A.ptr(), A.ld,
                   B.ptr(), B.ld, beta, C.ptr(), C.ld);
    } else {
        blas::symm(order(rowMajor), side, upLo, m, n, alpha, A.ptr(), A.ld,
                   B.ptr(), B.ld, beta, C.ptr(), C.ld);
    }

    std::string what = format("%s(%s,%s,%s,m=%d,n=%d,alpha=%s,beta=%s)",
                              hermitian ? "hemm" : "symm",
                              orderName(order(rowMajor)),
                              side==CblasLeft ? "L" : "R",
                              upLo==CblasLower ? "L" : "U", m, n,
                              str(alpha).c_str(), str(beta).c_str());
    return checkMatrix(ctx, C, ref, na, 'A', what)
        && ctx.check(C.paddingUnchanged(C0), "%s: padding of C modified",
                     what.c_str());
}

template <typename T>
void
testSymm(Context &ctx, bool hermitian)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int o=0; o<2; ++o) {
        for (int sd=0; sd<2; ++sd) {
            for (int u=0; u<2; ++u) {
                for (int d=0; d<numDims; ++d) {
                    for (std::size_t a=0; a<s.size(); ++a) {
                        for (std::size_t b=0; b<s.size(); ++b) {
                            if (!symmCase(ctx, random, hermitian, o==1,
                                          sides[sd], upLos[u], dims[d].m,
                                          dims[d].n, s[a], s[b]))
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    symmCase(ctx, random, hermitian, false, CblasLeft, CblasLower, 401, 9,
             s[2], s[2])
    && symmCase(ctx, random, hermitian, true, CblasRight, CblasUpper, 7, 399,
                s[2], s[2]);
}

//
//  SYRK, HERK:   C = alpha*op(A)*op(A)^T + beta*C  (op(A)^H for HERK)
//  SYR2K, HER2K: C = alpha*op(A)*op(B)^T + alpha*op(B)*op(A)^T + beta*C
//                (HER2K: alpha*op(A)*op(B)^H + conj(alpha)*op(B)*op(A)^H)
//
//  Only the triangle upLo of C is updated.  For HERK and HER2K the imaginary
//  parts of the diagonal of C are assumed to be zero on entry and set to
//  zero on exit.
//
template <typename T>
bool
rankkCase(Context &ctx, Random &random, bool hermitian, bool rank2,
          bool rowMajor, CBLAS_UPLO upLo, CBLAS_TRANSPOSE trans, int n, int k,
          T alpha, T beta)
{
    bool       t = trans!=CblasNoTrans;
    Matrix<T>  A(t ? k : n, t ? n : k, rowMajor), B(A.m, A.n, rowMajor);
    Matrix<T>  C(n, n, rowMajor);

    A.randomize(random);
    B.randomize(random);
    C.randomize(random);
    if (beta==T(0)) {
        fillNan(C);
    }
    // If C is updated, the imaginary parts of its diagonal are assumed to
    // be zero.  Reference BLAS returns immediately without touching C if
    // alpha or k is zero and beta is one.
    bool quickReturn = n==0 || ((alpha==T(0) || k==0) && beta==T(1));
    if (hermitian) {
        for (int i=0; i<n && beta!=T(0); ++i) {
            C(i,i) = fromCdouble<T>(cdouble(std::real(C(i,i)),
                                            quickReturn ? 0 : 77));
        }
    }

    // op(A) and op(B) are n x k
    CBLAS_TRANSPOSE  tc  = hermitian ? CblasConjTrans : CblasTrans;
    Full             opA = full(A, t ? tc : CblasNoTrans);
    Full             opB = full(B, t ? tc : CblasNoTrans);
    Full             fC  = full(C);
    Full             opAh(k, n), opBh(k, n);

    for (int i=0; i<n; ++i) {
        for (int l=0; l<k; ++l) {
            opAh(l,i) = hermitian ? std::conj(opA(i,l)) : opA(i,l);
            opBh(l,i) = hermitian ? std::conj(opB(i,l)) : opB(i,l);
        }
        if (hermitian) {
            fC(i,i) = cdouble(fC(i,i).real(), 0);
        }
    }

    cdouble  a   = toCdouble(alpha);
    cdouble  b   = toCdouble(beta);
    Result   ref;

    if (!rank2) {
        ref = multiply(a, opA, opAh, b, fC);
    } else {
        Result r1 = multiply(a, opA, opBh, b, fC);
        Result r2 = multiply(hermitian ? std::conj(a) : a, opB, opAh,
                             cdouble(0), fC);
        ref = r1;
        for (std::size_t p=0; p<ref.value.a.size(); ++p) {
            ref.value.a[p] += r2.value.a[p];
            ref.bound.a[p] += r2.bound.a[p];
        }
    }

    Matrix<T>  C0 = C;
    CBLAS_ORDER o = order(rowMajor);

    if (hermitian && !rank2) {
        blas::herk(o, upLo, trans, n, k, realPart(alpha), A.ptr(), A.ld,
                   realPart(beta), C.ptr(), C.ld);
    } else if (hermitian) {
        blas::her2k(o, upLo, trans, n, k, alpha, A.ptr(), A.ld, B.ptr(),
                    B.ld, realPart(beta), C.ptr(), C.ld);
    } else if (!rank2) {
        blas::syrk(o, upLo, trans, n, k, alpha, A.ptr(), A.ld, beta,
                   C.ptr(), C.ld);
    } else {
        blas::syr2k(o, upLo, trans, n, k, alpha, A.ptr(), A.ld, B.ptr(),
                    B.ld, beta, C.ptr(), C.ld);
    }

    std::string what = format("%s(%s,%s,%c,n=%d,k=%d,alpha=%s,beta=%s)",
                              hermitian ? (rank2 ? "her2k" : "herk")
                                        : (rank2 ? "syr2k" : "syrk"),
                              orderName(o), upLo==CblasLower ? "L" : "U",
                              transChar(trans), n, k, str(alpha).c_str(),
                              str(beta).c_str());
    char part = (upLo==CblasLower) ? 'L' : 'U';

    if (!checkMatrix(ctx, C, ref, rank2 ? 2*k : k, part, what)) {
        return false;
    }
    // the other strict triangle and the padding are not modified
    for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
            bool other = (upLo==CblasLower) ? i<j : i>j;
            if (other && !sameBits(C(i,j), C0(i,j))) {
                return ctx.check(false, "%s: element (%d,%d) outside of "
                                 "triangle modified", what.c_str(), i, j);
            }
        }
    }
    if (hermitian && !quickReturn) {
        for (int i=0; i<n; ++i) {
            if (std::imag(C(i,i))!=0) {
                return ctx.check(false, "%s: imaginary part of diagonal "
                                 "element %d not zero", what.c_str(), i);
            }
        }
    }
    return ctx.check(C.paddingUnchanged(C0), "%s: padding of C modified",
                     what.c_str());
}

template <typename T>
void
testRankk(Context &ctx, bool hermitian, bool rank2)
{
    Random          random;
    std::vector<T>  s = scalars<T>();
    CBLAS_TRANSPOSE t[2] = { CblasNoTrans,
                             hermitian ? CblasConjTrans : CblasTrans };

    for (int o=0; o<2; ++o) {
        for (int u=0; u<2; ++u) {
            for (int tr=0; tr<2; ++tr) {
                for (int d=0; d<numDims; ++d) {
                    for (std::size_t a=0; a<s.size(); ++a) {
                        for (std::size_t b=0; b<s.size(); ++b) {
                            T alpha = s[a], beta = s[b];
                            if (hermitian) {
                                beta = T(realPart(beta));
                                if (!rank2) {
                                    alpha = T(realPart(alpha));
                                }
                            }
                            if (!rankkCase(ctx, random, hermitian, rank2,
                                           o==1, upLos[u], t[tr], dims[d].n,
                                           dims[d].k, alpha, beta))
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    T alpha = hermitian && !rank2 ? T(realPart(s[2])) : s[2];
    T beta  = hermitian ? T(realPart(s[2])) : s[2];
    rankkCase(ctx, random, hermitian, rank2, false, CblasLower, t[1],
              397, 401, alpha, beta)
    && rankkCase(ctx, random, hermitian, rank2, true, CblasUpper, t[0],
                 389, 23, alpha, beta);
}

//
//  Triangular matrix for TRMM/TRSM/TRMV/TRSV.  For solvers the off-diagonal
//  elements are scaled and the diagonal is dominant (well conditioned).
//
template <typename T>
void
triangular(Matrix<T> &A, Random &random, CBLAS_UPLO upLo, CBLAS_DIAG diag,
           bool solve)
{
    int n = A.n;
    A.randomize(random);
    if (solve) {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<n; ++i) {
                A(i,j) = (i==j) ? A(i,j) + T(std::real(A(i,j))>=0 ? 2 : -2)
                                : A(i,j) / T(n);
            }
        }
    }
    poisonTriangle(A, upLo, diag==CblasUnit);
}

//
//  TRMM: B = alpha*op(A)*B (side left) or B = alpha*B*op(A) (side right)
//  TRSM: solve op(A)*X = alpha*B or X*op(A) = alpha*B, X overwrites B.
//        Checked by the residual |op(A)*X - alpha*B| <= tol*(|op(A)||X| +
//        |alpha||B|) (triangular solvers are backward stable).
//
template <typename T>
bool
trmmCase(Context &ctx, Random &random, bool solve, bool rowMajor,
         CBLAS_SIDE side, CBLAS_UPLO upLo, CBLAS_TRANSPOSE trans,
         CBLAS_DIAG diag, int m, int n, T alpha)
{
    int        na = (side==CblasLeft) ? m : n;
    Matrix<T>  A(na, na, rowMajor), B(m, n, rowMajor);

    triangular(A, random, upLo, diag, solve);
    B.randomize(random);
    if (alpha==T(0)) {
        fillNan(A);
        fillNan(B);
    }

    Full       opA = fullTriangular(A, upLo, diag, trans);
    Full       fB  = full(B);
    Matrix<T>  B0  = B;

    CBLAS_ORDER o = order(rowMajor);
    if (solve) {
        blas::trsm(o, side, upLo, trans, diag, m, n, alpha, A.ptr(), A.ld,
                   B.ptr(), B.ld);
    } else {
        blas::trmm(o, side, upLo, trans, diag, m, n, alpha, A.ptr(), A.ld,
                   B.ptr(), B.ld);
    }

    std::string what = format("%s(%s,%s,%s,%c,%s,m=%d,n=%d,alpha=%s)",
                              solve ? "trsm" : "trmm", orderName(o),
                              side==CblasLeft ? "L" : "R",
                              upLo==CblasLower ? "L" : "U", transChar(trans),
                              diag==CblasUnit ? "U" : "N", m, n,
                              str(alpha).c_str());

    if (!solve) {
        Result ref = (side==CblasLeft)
                   ? multiply(toCdouble(alpha), opA, fB, 0, fB)
                   : multiply(toCdouble(alpha), fB, opA, 0, fB);
        if (!checkMatrix(ctx, B, ref, na, 'A', what)) {
            return false;
        }
    } else if (alpha==T(0)) {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                if (B(i,j)!=T(0)) {
                    return ctx.check(false, "%s: element (%d,%d) not zero",
                                     what.c_str(), i, j);
                }
            }
        }
    } else {
        Full    X   = full(B);
        Result  res = (side==CblasLeft)
                    ? multiply(1, opA, X, -toCdouble(alpha), fB)
                    : multiply(1, X, opA, -toCdouble(alpha), fB);
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                if (!isFinite(B(i,j))
                 || std::abs(res.value(i,j))
                    > tolerance<T>(na)*res.bound(i,j).real())
                {
                    return ctx.check(false, "%s: residual (%d,%d) is %g, "
                                     "bound %g, %s", what.c_str(), i, j,
                                     std::abs(res.value(i,j)),
                                     res.bound(i,j).real(), typeName<T>());
                }
            }
        }
    }
    return ctx.check(B.paddingUnchanged(B0), "%s: padding of B modified",
                     what.c_str());
}

template <typename T>
void
testTrmm(Context &ctx, bool solve)
{
    Random          random;
    std::vector<T>  s = scalars<T>();

    for (int o=0; o<2; ++o) {
        for (int sd=0; sd<2; ++sd) {
            for (int u=0; u<2; ++u) {
                for (int t=0; t<3; ++t) {
                    for (int dg=0; dg<2; ++dg) {
                        for (int d=0; d<numDims; ++d) {
                            for (std::size_t a=0; a<s.size(); ++a) {
                                if (!trmmCase(ctx, random, solve, o==1,
                                              sides[sd], upLos[u],
                                              transposes[t], diags[dg],
                                              dims[d].m, dims[d].n, s[a]))
                                {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    trmmCase(ctx, random, solve, false, CblasLeft, CblasLower, CblasNoTrans,
             CblasNonUnit, 397, 11, s[2])
    && trmmCase(ctx, random, solve, true, CblasRight, CblasUpper,
                CblasConjTrans, CblasUnit, 9, 403, s[2]);
}

} // namespace

ULMBLAS_TEST(T0300, "level3/sgemm")  { testGemm<float>(ctx); }
ULMBLAS_TEST(T0301, "level3/dgemm")  { testGemm<double>(ctx); }
ULMBLAS_TEST(T0302, "level3/cgemm")  { testGemm<cfloat>(ctx); }
ULMBLAS_TEST(T0303, "level3/zgemm")  { testGemm<cdouble>(ctx); }

ULMBLAS_TEST(T0304, "level3/sgemm large")  { testGemmLarge<float>(ctx); }
ULMBLAS_TEST(T0305, "level3/dgemm large")  { testGemmLarge<double>(ctx); }
ULMBLAS_TEST(T0306, "level3/cgemm large")  { testGemmLarge<cfloat>(ctx); }
ULMBLAS_TEST(T0307, "level3/zgemm large")  { testGemmLarge<cdouble>(ctx); }

ULMBLAS_TEST(T0308, "level3/ssymm")  { testSymm<float>(ctx, false); }
ULMBLAS_TEST(T0309, "level3/dsymm")  { testSymm<double>(ctx, false); }
ULMBLAS_TEST(T0310, "level3/csymm")  { testSymm<cfloat>(ctx, false); }
ULMBLAS_TEST(T0311, "level3/zsymm")  { testSymm<cdouble>(ctx, false); }
ULMBLAS_TEST(T0312, "level3/chemm")  { testSymm<cfloat>(ctx, true); }
ULMBLAS_TEST(T0313, "level3/zhemm")  { testSymm<cdouble>(ctx, true); }

ULMBLAS_TEST(T0314, "level3/ssyrk")  { testRankk<float>(ctx, false, false); }
ULMBLAS_TEST(T0315, "level3/dsyrk")  { testRankk<double>(ctx, false, false); }
ULMBLAS_TEST(T0316, "level3/csyrk")  { testRankk<cfloat>(ctx, false, false); }
ULMBLAS_TEST(T0317, "level3/zsyrk")  { testRankk<cdouble>(ctx, false, false); }
ULMBLAS_TEST(T0318, "level3/cherk")  { testRankk<cfloat>(ctx, true, false); }
ULMBLAS_TEST(T0319, "level3/zherk")  { testRankk<cdouble>(ctx, true, false); }

ULMBLAS_TEST(T0320, "level3/ssyr2k") { testRankk<float>(ctx, false, true); }
ULMBLAS_TEST(T0321, "level3/dsyr2k") { testRankk<double>(ctx, false, true); }
ULMBLAS_TEST(T0322, "level3/csyr2k") { testRankk<cfloat>(ctx, false, true); }
ULMBLAS_TEST(T0323, "level3/zsyr2k") { testRankk<cdouble>(ctx, false, true); }
ULMBLAS_TEST(T0324, "level3/cher2k") { testRankk<cfloat>(ctx, true, true); }
ULMBLAS_TEST(T0325, "level3/zher2k") { testRankk<cdouble>(ctx, true, true); }

ULMBLAS_TEST(T0326, "level3/strmm")  { testTrmm<float>(ctx, false); }
ULMBLAS_TEST(T0327, "level3/dtrmm")  { testTrmm<double>(ctx, false); }
ULMBLAS_TEST(T0328, "level3/ctrmm")  { testTrmm<cfloat>(ctx, false); }
ULMBLAS_TEST(T0329, "level3/ztrmm")  { testTrmm<cdouble>(ctx, false); }

ULMBLAS_TEST(T0330, "level3/strsm")  { testTrmm<float>(ctx, true); }
ULMBLAS_TEST(T0331, "level3/dtrsm")  { testTrmm<double>(ctx, true); }
ULMBLAS_TEST(T0332, "level3/ctrsm")  { testTrmm<cfloat>(ctx, true); }
ULMBLAS_TEST(T0333, "level3/ztrsm")  { testTrmm<cdouble>(ctx, true); }
