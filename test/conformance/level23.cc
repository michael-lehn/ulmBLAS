//
//  Differential test of the level 2 and level 3 BLAS: the same calls against
//  ulmBLAS and against the reference BLAS/CBLAS of LAPACK 3.12.1 (two
//  executables, outputs compared by compare.cc; see conformance.sh).  Every array lies in a
//  buffer with guard cells (sentinels); unreferenced parts of matrices (other
//  triangle, unit diagonal, imaginary parts of Hermitian diagonals) are NaN,
//  padding rows and band corners are sentinels.  Inputs multiplied by
//  alpha = 0 and outputs scaled by beta = 0 are NaN in some modes.  Every
//  buffer is printed after the call.
//
//
//  The program is linked once with ulmBLAS and once with the reference; the
//  declarations of ulmBLAS (cblas.h, f77blas.h) are used for both, so the
//  index type of the ulmBLAS build must be int.  Argument: a group (gemv,
//  gbmv, symv, ger, syr, trv, gemm, symm, syrk, trm) or all.
//
#include <cblas.h>
#include <f77blas.h>
#include <cmath>
#include <cstdarg>
#include <algorithm>
#include <complex>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

typedef std::complex<float>  cf;
typedef std::complex<double> cd;

template <class T> struct RealT                  { typedef T type; };
template <class T> struct RealT<std::complex<T>> { typedef T type; };
template <class T> using Re = typename RealT<T>::type;
template <class T> constexpr bool isCx() { return !std::is_same<T, Re<T>>::value; }

template <class T> Re<T> *P(T *p)             { return reinterpret_cast<Re<T> *>(p); }
template <class T> const Re<T> *P(const T *p) { return reinterpret_cast<const Re<T> *>(p); }

inline float        sc(const float &a)  { return a; }
inline double       sc(const double &a) { return a; }
inline const float  *sc(const cf &a)     { return reinterpret_cast<const float *>(&a); }
inline const double *sc(const cd &a)     { return reinterpret_cast<const double *>(&a); }

template <class T> T *nc(const T *p) { return const_cast<T *>(p); }

template <class T> T mk(double re, double im)
{
    if constexpr (isCx<T>()) { return T(Re<T>(re), Re<T>(im)); }
    else                     { (void) im; return T(re); }
}

template <class T> T nanv()           { return mk<T>(NAN, NAN); }
template <class T> T val(int i, int j, int seed)
{
    return mk<T>((i*7 + j*3 + seed*5) % 7 - 3, (i*5 + j*2 + seed*3) % 5 - 2);
}

const int G = 3;                                   // guard cells per side
static int sentinelCount = 0;

//
//  Buffer with guard cells; data() points to the start of the storage.
//
template <class T>
struct Buf
{
    std::vector<T> v;
    explicit Buf(int len) : v(len + 2*G)
    {
        for (auto &x : v) { x = mk<T>(9000 + (sentinelCount++ % 997), 0); }
    }
    T *data() { return v.data() + G; }
};

template <class T>
void
print(Buf<T> &b)
{
    std::printf(" |");
    for (auto &x : b.v) {
        if constexpr (isCx<T>()) { std::printf(" %.9g,%.9g", (double) x.real(), (double) x.imag()); }
        else                     { std::printf(" %.9g", (double) x); }
    }
}

//
//  Vectors: logical element e of x with increment inc (BLAS convention).
//
inline int vlen(int n, int inc) { return n<=0 ? 1 : 1 + (n-1)*std::abs(inc); }
inline int vpos(int e, int n, int inc) { return inc>=0 ? e*inc : (n-1-e)*(-inc); }

template <class T>
Buf<T>
vec(int n, int inc, int seed, bool nan)
{
    Buf<T> b(vlen(n, inc));
    for (int e=0; e<n; ++e) {
        b.data()[vpos(e, n, inc)] = nan ? nanv<T>() : val<T>(e, 0, seed);
    }
    return b;
}

//
//  Matrix storage kinds.  kind: 'G' general, 'S' symmetric, 'H' Hermitian,
//  'T' triangular (unit: diag not referenced).  fmt: 'F' full, 'B' band,
//  'P' packed.  For 'G' band: kl, ku; for the others band: k.
//  Returns the buffer; NaN for unreferenced logical elements; if nan, the
//  referenced elements are NaN, too.  diagShift makes the diagonal of
//  triangular matrices non-zero.
//
struct MatSpec
{
    char kind = 'G', fmt = 'F', uplo = 'U', order = 'C';   // order: 'C' col, 'R' row
    bool unit = false;
    int  m = 0, n = 0, ld = 1, kl = 0, ku = 0, k = 0;
};

inline int bufLen(const MatSpec &s)
{
    if (s.fmt=='P') return std::max(1, s.n*(s.n+1)/2);
    int majors = s.fmt=='B' ? (s.order=='C' ? s.n : (s.kind=='G' ? s.m : s.n))
                            : (s.order=='C' ? s.n : s.m);
    return std::max(1, s.ld*std::max(majors, 1));
}

// storage position of logical (i,j) or -1 if not stored
inline int spos(const MatSpec &s, int i, int j)
{
    bool col = s.order=='C';
    if (s.fmt=='F') return col ? i + j*s.ld : i*s.ld + j;
    if (s.fmt=='B') {
        if (s.kind=='G') {
            if (i-j > s.kl || j-i > s.ku) return -1;
            return col ? (s.ku+i-j) + j*s.ld : i*s.ld + (s.kl+j-i);
        }
        if (s.uplo=='U') {
            if (i>j || j-i > s.k) return -1;
            return col ? (s.k+i-j) + j*s.ld : i*s.ld + (j-i);
        }
        if (i<j || i-j > s.k) return -1;
        return col ? (i-j) + j*s.ld : i*s.ld + (s.k+j-i);
    }
    // packed
    int n = s.n;
    if (s.uplo=='U') {
        if (i>j) return -1;
        return col ? i + j*(j+1)/2 : i*n - i*(i-1)/2 + (j-i);
    }
    if (i<j) return -1;
    return col ? (i-j) + j*(2*n-j+1)/2 : i*(i+1)/2 + j;
}

template <class T>
Buf<T>
mat(const MatSpec &s, int seed, bool nan)
{
    Buf<T> b(bufLen(s));
    T *A = b.data();
    int rows = s.kind=='G' ? s.m : s.n;
    for (int j=0; j<s.n; ++j) {
        for (int i=0; i<rows; ++i) {
            bool tri  = s.kind!='G';
            bool ref  = !tri || (s.uplo=='U' ? i<=j : i>=j);
            int  p    = spos(s, i, j);
            if (p<0) {
                // other triangle of a full matrix: NaN (not referenced)
                if (s.fmt=='F' && tri) {
                    p = s.order=='C' ? i + j*s.ld : i*s.ld + j;
                    A[p] = nanv<T>();
                }
                continue;
            }
            if (!ref) { A[p] = nanv<T>(); continue; }
            T x = nan ? nanv<T>() : val<T>(i, j, seed);
            if (i==j && s.kind=='T') {
                x = s.unit ? nanv<T>() : (nan ? x : mk<T>(2 + (i%2), (i%3) - 1));
            }
            if (i==j && s.kind=='H') {
                x = mk<T>(std::real(x), NAN);        // imaginary part not referenced
            }
            A[p] = x;
        }
    }
    return b;
}

//
//  Output
//
static std::string current;
inline void head(const std::string &s) { current = s; std::printf("%s:", s.c_str()); }
inline void end()                      { std::printf("\n"); }

template <class... B> void printAll(B &...b) { (print(b), ...); }

inline std::string fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char *f, ...)
{
    char buf[512]; va_list ap; va_start(ap, f); std::vsnprintf(buf, sizeof buf, f, ap); va_end(ap);
    return buf;
}

template <class T> const char *tn() {
    return std::is_same<T,float>::value ? "s" : std::is_same<T,double>::value ? "d"
         : std::is_same<T,cf>::value ? "c" : "z";
}

//
//  Scalars: (alpha, beta, nanIn, nanOut) modes
//
struct Mode { double a_re, a_im, b_re, b_im; bool nanIn, nanOut; };
const Mode modes[] = {
    { 2, -1, 0.5, 0.5, false, false },
    { 0,  0, 0.5, 0,   true,  false },
    { 1,  0, 0,   0,   false, true  },
    { 0,  0, 1,   0,   true,  false },
    { 0,  0, 0,   0,   true,  true  },
    { 1,  0, 1,   0,   false, false },
};

static CBLAS_ORDER  co(char o) { return o=='C' ? CblasColMajor : CblasRowMajor; }
static CBLAS_TRANSPOSE ct(char t) { return t=='N' ? CblasNoTrans : t=='T' ? CblasTrans : CblasConjTrans; }
static CBLAS_UPLO   cu(char u) { return u=='U' ? CblasUpper : CblasLower; }
static CBLAS_DIAG   cdg(char d) { return d=='U' ? CblasUnit : CblasNonUnit; }
static CBLAS_SIDE   cs(char s) { return s=='L' ? CblasLeft : CblasRight; }

#define PICK(T, NAME) \
    (std::is_same<T,float>::value ? (void *) s##NAME##_ : std::is_same<T,double>::value \
     ? (void *) d##NAME##_ : std::is_same<T,cf>::value ? (void *) c##NAME##_ : (void *) z##NAME##_)

// call the typed CBLAS function: real by s/d prefix, complex by c/z prefix
#define CB4(T, NAME, ...)                                                   \
    do {                                                                    \
        if constexpr (std::is_same<T,float>::value)  cblas_s##NAME(__VA_ARGS__); \
        else if constexpr (std::is_same<T,double>::value) cblas_d##NAME(__VA_ARGS__); \
        else if constexpr (std::is_same<T,cf>::value) cblas_c##NAME(__VA_ARGS__); \
        else cblas_z##NAME(__VA_ARGS__);                                    \
    } while (0)
#define F4(T, NAME, ...)                                                    \
    do {                                                                    \
        if constexpr (std::is_same<T,float>::value)  s##NAME##_(__VA_ARGS__); \
        else if constexpr (std::is_same<T,double>::value) d##NAME##_(__VA_ARGS__); \
        else if constexpr (std::is_same<T,cf>::value) c##NAME##_(__VA_ARGS__); \
        else z##NAME##_(__VA_ARGS__);                                       \
    } while (0)
// real: s/d NAMER, complex: c/z NAMEC
#define CB2(T, NAMER, NAMEC, ...)                                           \
    do {                                                                    \
        if constexpr (std::is_same<T,float>::value)  cblas_s##NAMER(__VA_ARGS__); \
        else if constexpr (std::is_same<T,double>::value) cblas_d##NAMER(__VA_ARGS__); \
        else if constexpr (std::is_same<T,cf>::value) cblas_c##NAMEC(__VA_ARGS__); \
        else cblas_z##NAMEC(__VA_ARGS__);                                   \
    } while (0)
#define F2(T, NAMER, NAMEC, ...)                                            \
    do {                                                                    \
        if constexpr (std::is_same<T,float>::value)  s##NAMER##_(__VA_ARGS__); \
        else if constexpr (std::is_same<T,double>::value) d##NAMER##_(__VA_ARGS__); \
        else if constexpr (std::is_same<T,cf>::value) c##NAMEC##_(__VA_ARGS__); \
        else z##NAMEC##_(__VA_ARGS__);                                      \
    } while (0)

const int dims[] = { 0, 1, 5 };
const int vincs[][2] = { {1, 1}, {-1, 2}, {2, -2} };
const char ifaces[] = { 'F', 'C', 'R' };      // F77, CBLAS col major, CBLAS row major

//
//  ---------------------------------------------------------------- gemv
//
template <class T>
void
gemv()
{
    for (char I : ifaces) for (char t : {'N','T','C'}) for (int m : dims) for (int n : dims)
    for (int pad : {0, 2}) for (auto &ic : vincs) for (const Mode &md : modes) {
        // reference CBLAS defect: c/z gemv, gbmv RowMajor ConjTrans with M==0,
        // N>0 negate y with uninitialized loop bounds (cblas_cgemv.c:147)
        if (isCx<T>() && I=='R' && t=='C' && m==0 && n>0) continue;
        MatSpec s; s.order = I=='R' ? 'R' : 'C'; s.m = m; s.n = n;
        s.ld = std::max(1, (s.order=='C' ? m : n)) + pad;
        int lx = t=='N' ? n : m, ly = t=='N' ? m : n;
        auto A = mat<T>(s, 1, md.nanIn);
        auto x = vec<T>(lx, ic[0], 2, md.nanIn);
        auto y = vec<T>(ly, ic[1], 3, md.nanOut);
        T al = mk<T>(md.a_re, md.a_im), be = mk<T>(md.b_re, md.b_im);
        head(fmt("%sgemv %c t=%c m=%d n=%d ld=%d inc=%d,%d a=%g,%g b=%g,%g", tn<T>(), I, t, m, n,
                 s.ld, ic[0], ic[1], md.a_re, md.a_im, md.b_re, md.b_im));
        int ix = ic[0], iy = ic[1], ld = s.ld;
        if (I=='F') F4(T, gemv, &t, &m, &n, P(&al), P(A.data()), &ld, P(x.data()), &ix, P(&be), P(y.data()), &iy);
        else CB4(T, gemv, co(s.order), ct(t), m, n, sc(al), P(A.data()), ld, P(x.data()), ix, sc(be), P(y.data()), iy);
        printAll(A, x, y); end();
    }
}

//
//  ---------------------------------------------------------------- gbmv
//
template <class T>
void
gbmv()
{
    for (char I : ifaces) for (char t : {'N','T','C'}) for (int m : dims) for (int n : dims)
    for (int kl : {0, 2}) for (int ku : {0, 1}) for (int pad : {0, 1}) for (auto &ic : vincs)
    for (const Mode &md : modes) {
        // reference CBLAS defect: c/z gemv, gbmv RowMajor ConjTrans with M==0,
        // N>0 negate y with uninitialized loop bounds (cblas_cgemv.c:147)
        if (isCx<T>() && I=='R' && t=='C' && m==0 && n>0) continue;
        MatSpec s; s.fmt = 'B'; s.order = I=='R' ? 'R' : 'C'; s.m = m; s.n = n; s.kl = kl; s.ku = ku;
        s.ld = kl + ku + 1 + pad;
        int lx = t=='N' ? n : m, ly = t=='N' ? m : n;
        auto A = mat<T>(s, 1, md.nanIn);
        auto x = vec<T>(lx, ic[0], 2, md.nanIn);
        auto y = vec<T>(ly, ic[1], 3, md.nanOut);
        T al = mk<T>(md.a_re, md.a_im), be = mk<T>(md.b_re, md.b_im);
        head(fmt("%sgbmv %c t=%c m=%d n=%d kl=%d ku=%d ld=%d inc=%d,%d a=%g,%g b=%g,%g", tn<T>(), I, t,
                 m, n, kl, ku, s.ld, ic[0], ic[1], md.a_re, md.a_im, md.b_re, md.b_im));
        int ix = ic[0], iy = ic[1], ld = s.ld;
        if (I=='F') F4(T, gbmv, &t, &m, &n, &kl, &ku, P(&al), P(A.data()), &ld, P(x.data()), &ix, P(&be), P(y.data()), &iy);
        else CB4(T, gbmv, co(s.order), ct(t), m, n, kl, ku, sc(al), P(A.data()), ld, P(x.data()), ix, sc(be), P(y.data()), iy);
        printAll(A, x, y); end();
    }
}

//
//  ------------------------------------------- symv/hemv, sbmv/hbmv, spmv/hpmv
//
template <class T>
void
symv(char fmtc)
{
    for (char I : ifaces) for (char u : {'U','L'}) for (int n : dims) for (int k : {0, 2})
    for (int pad : {0, 1}) for (auto &ic : vincs) for (const Mode &md : modes) {
        if (fmtc!='B' && k) continue;
        if (fmtc=='P' && pad) continue;
        MatSpec s; s.kind = isCx<T>() ? 'H' : 'S'; s.fmt = fmtc; s.uplo = u;
        s.order = I=='R' ? 'R' : 'C'; s.m = s.n = n; s.k = k;
        s.ld = (fmtc=='B' ? k+1 : std::max(1, n)) + pad;
        auto A = mat<T>(s, 1, md.nanIn);
        auto x = vec<T>(n, ic[0], 2, md.nanIn);
        auto y = vec<T>(n, ic[1], 3, md.nanOut);
        T al = mk<T>(md.a_re, md.a_im), be = mk<T>(md.b_re, md.b_im);
        head(fmt("%s%cmv %c u=%c n=%d k=%d ld=%d inc=%d,%d a=%g,%g b=%g,%g", tn<T>(), fmtc, I, u, n, k,
                 s.ld, ic[0], ic[1], md.a_re, md.a_im, md.b_re, md.b_im));
        int ix = ic[0], iy = ic[1], ld = s.ld;
        CBLAS_ORDER o = co(s.order);
        if (fmtc=='F') {
            if (I=='F') F2(T, symv, hemv, &u, &n, P(&al), P(A.data()), &ld, P(x.data()), &ix, P(&be), P(y.data()), &iy);
            else CB2(T, symv, hemv, o, cu(u), n, sc(al), P(A.data()), ld, P(x.data()), ix, sc(be), P(y.data()), iy);
        } else if (fmtc=='B') {
            if (I=='F') F2(T, sbmv, hbmv, &u, &n, &k, P(&al), P(A.data()), &ld, P(x.data()), &ix, P(&be), P(y.data()), &iy);
            else CB2(T, sbmv, hbmv, o, cu(u), n, k, sc(al), P(A.data()), ld, P(x.data()), ix, sc(be), P(y.data()), iy);
        } else {
            if (I=='F') F2(T, spmv, hpmv, &u, &n, P(&al), P(A.data()), P(x.data()), &ix, P(&be), P(y.data()), &iy);
            else CB2(T, spmv, hpmv, o, cu(u), n, sc(al), P(A.data()), P(x.data()), ix, sc(be), P(y.data()), iy);
        }
        printAll(A, x, y); end();
    }
}

//
//  ---------------------------------------------------------------- ger, geru, gerc
//
template <class T>
void
ger(int variant)                       // 0: ger/geru, 1: gerc
{
    if (!isCx<T>() && variant==1) return;
    for (char I : ifaces) for (int m : dims) for (int n : dims) for (int pad : {0, 2})
    for (auto &ic : vincs) for (int mi : {0, 1, 5}) {
        const Mode &md = modes[mi];
        MatSpec s; s.order = I=='R' ? 'R' : 'C'; s.m = m; s.n = n;
        s.ld = std::max(1, (s.order=='C' ? m : n)) + pad;
        auto A = mat<T>(s, 1, false);
        auto x = vec<T>(m, ic[0], 2, md.nanIn);
        auto y = vec<T>(n, ic[1], 3, md.nanIn);
        T al = mk<T>(md.a_re, md.a_im);
        head(fmt("%sger%s %c m=%d n=%d ld=%d inc=%d,%d a=%g,%g", tn<T>(), variant ? "c" : "u", I, m, n,
                 s.ld, ic[0], ic[1], md.a_re, md.a_im));
        int ix = ic[0], iy = ic[1], ld = s.ld;
        if (variant==0) {
            if (I=='F') F2(T, ger, geru, &m, &n, P(&al), P(x.data()), &ix, P(y.data()), &iy, P(A.data()), &ld);
            else CB2(T, ger, geru, co(s.order), m, n, sc(al), P(x.data()), ix, P(y.data()), iy, P(A.data()), ld);
        } else {
            if constexpr (isCx<T>()) {
                if (I=='F') F2(T, ger, gerc, &m, &n, P(&al), P(x.data()), &ix, P(y.data()), &iy, P(A.data()), &ld);
                else CB2(T, ger, gerc, co(s.order), m, n, sc(al), P(x.data()), ix, P(y.data()), iy, P(A.data()), ld);
            }
        }
        printAll(A, x, y); end();
    }
}

//
//  ------------------------------------------- syr/her, spr/hpr, syr2/her2, spr2/hpr2
//
template <class T>
void
syr(char fmtc, bool two)
{
    for (char I : ifaces) for (char u : {'U','L'}) for (int n : dims) for (int pad : {0, 1})
    for (auto &ic : vincs) for (int mi : {0, 1, 5}) {
        if (fmtc=='P' && pad) continue;
        const Mode &md = modes[mi];
        MatSpec s; s.kind = isCx<T>() ? 'H' : 'S'; s.fmt = fmtc; s.uplo = u;
        s.order = I=='R' ? 'R' : 'C'; s.m = s.n = n; s.ld = std::max(1, n) + pad;
        auto A = mat<T>(s, 1, false);
        auto x = vec<T>(n, ic[0], 2, md.nanIn);
        auto y = vec<T>(n, ic[1], 3, md.nanIn);
        head(fmt("%s%cr%s %c u=%c n=%d ld=%d inc=%d,%d a=%g,%g", tn<T>(), fmtc, two ? "2" : "", I, u, n,
                 s.ld, ic[0], ic[1], md.a_re, md.a_im));
        int ix = ic[0], iy = ic[1], ld = s.ld;
        CBLAS_ORDER o = co(s.order);
        Re<T> ar = Re<T>(md.a_re);                   // syr, spr, her, hpr: real alpha
        T al = mk<T>(md.a_re, md.a_im);
        if (!two && fmtc=='F') {
            if (I=='F') F2(T, syr, her, &u, &n, &ar, P(x.data()), &ix, P(A.data()), &ld);
            else CB2(T, syr, her, o, cu(u), n, ar, P(x.data()), ix, P(A.data()), ld);
        } else if (!two) {
            if (I=='F') F2(T, spr, hpr, &u, &n, &ar, P(x.data()), &ix, P(A.data()));
            else CB2(T, spr, hpr, o, cu(u), n, ar, P(x.data()), ix, P(A.data()));
        } else if (fmtc=='F') {
            if (I=='F') F2(T, syr2, her2, &u, &n, P(&al), P(x.data()), &ix, P(y.data()), &iy, P(A.data()), &ld);
            else CB2(T, syr2, her2, o, cu(u), n, sc(al), P(x.data()), ix, P(y.data()), iy, P(A.data()), ld);
        } else {
            if (I=='F') F2(T, spr2, hpr2, &u, &n, P(&al), P(x.data()), &ix, P(y.data()), &iy, P(A.data()));
            else CB2(T, spr2, hpr2, o, cu(u), n, sc(al), P(x.data()), ix, P(y.data()), iy, P(A.data()));
        }
        printAll(A, x, y); end();
    }
}

//
//  ------------------------------------------- trmv, tbmv, tpmv, trsv, tbsv, tpsv
//
template <class T>
void
trv(char fmtc, bool solve)
{
    for (char I : ifaces) for (char u : {'U','L'}) for (char t : {'N','T','C'}) for (char d : {'N','U'})
    for (int n : dims) for (int k : {0, 2}) for (int pad : {0, 1}) for (int inc : {1, -1, 2, -2})
    for (bool zeroX : {false, true}) {
        if (fmtc!='B' && k) continue;
        if (fmtc=='P' && pad) continue;
        MatSpec s; s.kind = 'T'; s.fmt = fmtc; s.uplo = u; s.unit = d=='U';
        s.order = I=='R' ? 'R' : 'C'; s.m = s.n = n; s.k = k;
        s.ld = (fmtc=='B' ? k+1 : std::max(1, n)) + pad;
        auto A = mat<T>(s, 1, false);
        auto x = vec<T>(n, inc, 2, false);
        if (zeroX && n>0) {
            // x = 0 except the last element; and NaN in one column of A
            for (int e=0; e<n; ++e) x.data()[vpos(e, n, inc)] = e==n-1 ? mk<T>(1, 1) : T(0);
            int j = 0;
            for (int i=0; i<n; ++i) {
                int p = spos(s, i, j);
                if (p>=0 && i!=j && (u=='U' ? i<=j : i>=j)) A.data()[p] = nanv<T>();
            }
        }
        head(fmt("%st%c%s %c u=%c t=%c d=%c n=%d k=%d ld=%d inc=%d zx=%d", tn<T>(), fmtc, solve ? "sv" : "mv",
                 I, u, t, d, n, k, s.ld, inc, (int) zeroX));
        int ld = s.ld;
        CBLAS_ORDER o = co(s.order);
        if (fmtc=='F' && !solve) {
            if (I=='F') F4(T, trmv, &u, &t, &d, &n, P(A.data()), &ld, P(x.data()), &inc);
            else CB4(T, trmv, o, cu(u), ct(t), cdg(d), n, P(A.data()), ld, P(x.data()), inc);
        } else if (fmtc=='F') {
            if (I=='F') F4(T, trsv, &u, &t, &d, &n, P(A.data()), &ld, P(x.data()), &inc);
            else CB4(T, trsv, o, cu(u), ct(t), cdg(d), n, P(A.data()), ld, P(x.data()), inc);
        } else if (fmtc=='B' && !solve) {
            if (I=='F') F4(T, tbmv, &u, &t, &d, &n, &k, P(A.data()), &ld, P(x.data()), &inc);
            else CB4(T, tbmv, o, cu(u), ct(t), cdg(d), n, k, P(A.data()), ld, P(x.data()), inc);
        } else if (fmtc=='B') {
            if (I=='F') F4(T, tbsv, &u, &t, &d, &n, &k, P(A.data()), &ld, P(x.data()), &inc);
            else CB4(T, tbsv, o, cu(u), ct(t), cdg(d), n, k, P(A.data()), ld, P(x.data()), inc);
        } else if (!solve) {
            if (I=='F') F4(T, tpmv, &u, &t, &d, &n, P(A.data()), P(x.data()), &inc);
            else CB4(T, tpmv, o, cu(u), ct(t), cdg(d), n, P(A.data()), P(x.data()), inc);
        } else {
            if (I=='F') F4(T, tpsv, &u, &t, &d, &n, P(A.data()), P(x.data()), &inc);
            else CB4(T, tpsv, o, cu(u), ct(t), cdg(d), n, P(A.data()), P(x.data()), inc);
        }
        printAll(A, x); end();
    }
}

//
//  ---------------------------------------------------------------- level 3
//
const int dims3[] = { 0, 1, 5, 9 };

template <class T>
void
gemm()
{
    for (char I : ifaces) for (char ta : {'N','T','C'}) for (char tb : {'N','T','C'})
    for (int m : dims3) for (int n : dims3) for (int k : dims3) for (int pad : {0, 1})
    for (const Mode &md : modes) {
        if (!(pad==0 || (m==5 && n==5 && k==5))) continue;
        char o = I=='R' ? 'R' : 'C';
        MatSpec sa; sa.order = o; sa.m = ta=='N' ? m : k; sa.n = ta=='N' ? k : m;
        sa.ld = std::max(1, o=='C' ? sa.m : sa.n) + pad;
        MatSpec sb; sb.order = o; sb.m = tb=='N' ? k : n; sb.n = tb=='N' ? n : k;
        sb.ld = std::max(1, o=='C' ? sb.m : sb.n) + 2*pad;
        MatSpec sc_; sc_.order = o; sc_.m = m; sc_.n = n; sc_.ld = std::max(1, o=='C' ? m : n) + 3*pad;
        auto A = mat<T>(sa, 1, md.nanIn), B = mat<T>(sb, 2, md.nanIn), C = mat<T>(sc_, 3, md.nanOut);
        T al = mk<T>(md.a_re, md.a_im), be = mk<T>(md.b_re, md.b_im);
        head(fmt("%sgemm %c ta=%c tb=%c m=%d n=%d k=%d pad=%d a=%g,%g b=%g,%g", tn<T>(), I, ta, tb, m, n, k,
                 pad, md.a_re, md.a_im, md.b_re, md.b_im));
        int la = sa.ld, lb = sb.ld, lc = sc_.ld;
        if (I=='F') F4(T, gemm, &ta, &tb, &m, &n, &k, P(&al), P(A.data()), &la, P(B.data()), &lb, P(&be), P(C.data()), &lc);
        else CB4(T, gemm, co(o), ct(ta), ct(tb), m, n, k, sc(al), P(A.data()), la, P(B.data()), lb, sc(be), P(C.data()), lc);
        printAll(A, B, C); end();
    }
}

template <class T>
void
symm(bool herm)
{
    if (herm && !isCx<T>()) return;
    for (char I : ifaces) for (char sd : {'L','R'}) for (char u : {'U','L'})
    for (int m : dims3) for (int n : dims3) for (int pad : {0, 1}) for (const Mode &md : modes) {
        if (!(pad==0 || (m==5 && n==5))) continue;
        char o = I=='R' ? 'R' : 'C';
        int na = sd=='L' ? m : n;
        MatSpec sa; sa.kind = herm ? 'H' : 'S'; sa.uplo = u; sa.order = o; sa.m = sa.n = na;
        sa.ld = std::max(1, na) + pad;
        MatSpec sb; sb.order = o; sb.m = m; sb.n = n; sb.ld = std::max(1, o=='C' ? m : n) + 2*pad;
        MatSpec sc_ = sb; sc_.ld += pad;
        auto A = mat<T>(sa, 1, md.nanIn), B = mat<T>(sb, 2, md.nanIn), C = mat<T>(sc_, 3, md.nanOut);
        T al = mk<T>(md.a_re, md.a_im), be = mk<T>(md.b_re, md.b_im);
        head(fmt("%s%s %c s=%c u=%c m=%d n=%d pad=%d a=%g,%g b=%g,%g", tn<T>(), herm ? "hemm" : "symm", I, sd, u,
                 m, n, pad, md.a_re, md.a_im, md.b_re, md.b_im));
        int la = sa.ld, lb = sb.ld, lc = sc_.ld;
        if (!herm) {
            if (I=='F') F4(T, symm, &sd, &u, &m, &n, P(&al), P(A.data()), &la, P(B.data()), &lb, P(&be), P(C.data()), &lc);
            else CB4(T, symm, co(o), cs(sd), cu(u), m, n, sc(al), P(A.data()), la, P(B.data()), lb, sc(be), P(C.data()), lc);
        } else if constexpr (isCx<T>()) {
            if (I=='F') F2(T, symm, hemm, &sd, &u, &m, &n, P(&al), P(A.data()), &la, P(B.data()), &lb, P(&be), P(C.data()), &lc);
            else CB2(T, symm, hemm, co(o), cs(sd), cu(u), m, n, nc(sc(al)), P(A.data()), la, P(B.data()), lb, nc(sc(be)), P(C.data()), lc);
        }
        printAll(A, B, C); end();
    }
}

template <class T>
void
syrk(bool herm, bool two)
{
    if (herm && !isCx<T>()) return;
    for (char I : ifaces) for (char u : {'U','L'}) for (char t : {'N','T','C'})
    for (int n : dims3) for (int k : dims3) for (int pad : {0, 1}) for (const Mode &md : modes) {
        if (!(pad==0 || (n==5 && k==5))) continue;
        if (herm && t=='T') continue;           // herk: N or C
        if (!herm && t=='C' && isCx<T>()) continue;   // csyrk: N or T
        char o = I=='R' ? 'R' : 'C';
        MatSpec sa; sa.order = o; sa.m = t=='N' ? n : k; sa.n = t=='N' ? k : n;
        sa.ld = std::max(1, o=='C' ? sa.m : sa.n) + pad;
        MatSpec sc_; sc_.kind = herm ? 'H' : 'S'; sc_.uplo = u; sc_.order = o; sc_.m = sc_.n = n;
        sc_.ld = std::max(1, n) + 3*pad;
        MatSpec sb = sa; sb.ld += pad;
        auto A = mat<T>(sa, 1, md.nanIn), B = mat<T>(sb, 2, md.nanIn), C = mat<T>(sc_, 3, md.nanOut);
        if (herm) {        // C's diagonal imaginary parts: given (not NaN), herk sets them to 0
            for (int i=0; i<n; ++i) { T &c = C.data()[spos(sc_, i, i)]; c = mk<T>(std::real(c), 1); }
        }
        T al = mk<T>(md.a_re, md.a_im), be = mk<T>(md.b_re, md.b_im);
        Re<T> ar = Re<T>(md.a_re), br = Re<T>(md.b_re);
        head(fmt("%s%s %c u=%c t=%c n=%d k=%d pad=%d a=%g,%g b=%g,%g", tn<T>(),
                 herm ? (two ? "her2k" : "herk") : (two ? "syr2k" : "syrk"), I, u, t, n, k, pad,
                 md.a_re, md.a_im, md.b_re, md.b_im));
        int la = sa.ld, lb = sb.ld, lc = sc_.ld;
        CBLAS_ORDER co_ = co(o);
        if (!herm && !two) {
            if (I=='F') F4(T, syrk, &u, &t, &n, &k, P(&al), P(A.data()), &la, P(&be), P(C.data()), &lc);
            else CB4(T, syrk, co_, cu(u), ct(t), n, k, sc(al), P(A.data()), la, sc(be), P(C.data()), lc);
        } else if (!herm) {
            if (I=='F') F4(T, syr2k, &u, &t, &n, &k, P(&al), P(A.data()), &la, P(B.data()), &lb, P(&be), P(C.data()), &lc);
            else CB4(T, syr2k, co_, cu(u), ct(t), n, k, sc(al), P(A.data()), la, P(B.data()), lb, sc(be), P(C.data()), lc);
        } else if constexpr (isCx<T>()) {
            if (!two) {
                if (I=='F') F2(T, syrk, herk, &u, &t, &n, &k, &ar, P(A.data()), &la, &br, P(C.data()), &lc);
                else CB2(T, syrk, herk, co_, cu(u), ct(t), n, k, ar, P(A.data()), la, br, P(C.data()), lc);
            } else {
                if (I=='F') F2(T, syr2k, her2k, &u, &t, &n, &k, P(&al), P(A.data()), &la, P(B.data()), &lb, &br, P(C.data()), &lc);
                else CB2(T, syr2k, her2k, co_, cu(u), ct(t), n, k, nc(sc(al)), P(A.data()), la, P(B.data()), lb, br, P(C.data()), lc);
            }
        }
        printAll(A, B, C); end();
    }
}

template <class T>
void
trm(bool solve)
{
    for (char I : ifaces) for (char sd : {'L','R'}) for (char u : {'U','L'}) for (char t : {'N','T','C'})
    for (char d : {'N','U'}) for (int m : dims3) for (int n : dims3) for (int pad : {0, 1})
    for (int mi : {0, 1, 5}) {
        if (!(pad==0 || (m==5 && n==5))) continue;
        const Mode &md = modes[mi];
        char o = I=='R' ? 'R' : 'C';
        int na = sd=='L' ? m : n;
        MatSpec sa; sa.kind = 'T'; sa.uplo = u; sa.unit = d=='U'; sa.order = o; sa.m = sa.n = na;
        sa.ld = std::max(1, na) + pad;
        MatSpec sb; sb.order = o; sb.m = m; sb.n = n; sb.ld = std::max(1, o=='C' ? m : n) + 2*pad;
        auto A = mat<T>(sa, 1, false), B = mat<T>(sb, 2, md.nanIn);
        T al = mk<T>(md.a_re, md.a_im);
        head(fmt("%s%s %c s=%c u=%c t=%c d=%c m=%d n=%d pad=%d a=%g,%g", tn<T>(), solve ? "trsm" : "trmm", I, sd,
                 u, t, d, m, n, pad, md.a_re, md.a_im));
        int la = sa.ld, lb = sb.ld;
        if (!solve) {
            if (I=='F') F4(T, trmm, &sd, &u, &t, &d, &m, &n, P(&al), P(A.data()), &la, P(B.data()), &lb);
            else CB4(T, trmm, co(o), cs(sd), cu(u), ct(t), cdg(d), m, n, sc(al), P(A.data()), la, P(B.data()), lb);
        } else {
            if (I=='F') F4(T, trsm, &sd, &u, &t, &d, &m, &n, P(&al), P(A.data()), &la, P(B.data()), &lb);
            else CB4(T, trsm, co(o), cs(sd), cu(u), ct(t), cdg(d), m, n, sc(al), P(A.data()), la, P(B.data()), lb);
        }
        printAll(A, B); end();
    }
}

template <class T>
void
all(const std::string &which)
{
    auto want = [&](const char *g) { return which=="all" || which==g; };
    if (want("gemv")) gemv<T>();
    if (want("gbmv")) gbmv<T>();
    if (want("symv")) { symv<T>('F'); symv<T>('B'); symv<T>('P'); }
    if (want("ger"))  { ger<T>(0); ger<T>(1); }
    if (want("syr"))  { syr<T>('F', false); syr<T>('P', false); syr<T>('F', true); syr<T>('P', true); }
    if (want("trv"))  { for (char f : {'F','B','P'}) { trv<T>(f, false); trv<T>(f, true); } }
    if (want("gemm")) gemm<T>();
    if (want("symm")) { symm<T>(false); symm<T>(true); }
    if (want("syrk")) { syrk<T>(false, false); syrk<T>(false, true); syrk<T>(true, false); syrk<T>(true, true); }
    if (want("trm"))  { trm<T>(false); trm<T>(true); }
}

int
main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string which = argc>1 ? argv[1] : "all";
    all<float>(which); all<double>(which); all<cf>(which); all<cd>(which);
}
