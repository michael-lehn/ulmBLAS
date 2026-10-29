//
//  ulmBLAS: the definitions of BLIS that the vendored BLIS micro kernels in
//  kernel/*.h use (they are compiled as C++ inside ulmBLAS, without
//  blis.h).  Not part of BLIS; written for ulmBLAS after the definitions in
//  BLIS 2.0 (frame/include/bli_type_defs.h, bli_edge_case_macro_defs.h).
//
//  Types: dim_t and inc_t are 64-bit integers as in BLIS; auxinfo_t holds
//  the addresses of the next panels of A and B (the only fields the kernels
//  read); cntx_t is only passed through; scomplex and dcomplex as in BLIS
//  (layout of std::complex<float>, std::complex<double>).
//
//  Edge case macros (BLIS 1.0 and newer kernels): GEMM_UKR_SETUP_CT* decide
//  whether the kernel writes to a local column or row major micro tile
//  _ct instead of C (C stored with a stride the assembly does not support,
//  a partial tile, or for ..._ALIGNED a misaligned C) and then let the
//  kernel compute _ct = alpha*A*B; GEMM_UKR_FLUSH_CT updates
//  C <- beta*C + _ct in C++ (C is not read if beta = 0).  ulmBLAS passes
//  full tiles only, so the local tile is used only for the storage of C.
//
//  The macros are undefined again by compat_end.h at the end of every
//  kernel header.
//
#ifndef ULMBLAS_LEVEL3_UKERNEL_BLIS_COMPAT_TYPES
#define ULMBLAS_LEVEL3_UKERNEL_BLIS_COMPAT_TYPES 1

#include <cstdint>

namespace ulmBLAS { namespace blis { namespace compat {

typedef std::int64_t    dim_t;
typedef std::int64_t    inc_t;

struct auxinfo_t
{
    const void *a_next;
    const void *b_next;
};

struct cntx_t;

//  BLIS: complex types (real and imaginary part, the layout of
//  std::complex<float> and std::complex<double>)
struct scomplex
{
    float   real, imag;
};

struct dcomplex
{
    double  real, imag;
};

//  AOCL-BLAS (frame/base/bli_const.h): classification of alpha and beta used
//  by some AOCL kernels
enum mulfactor { BLIS_MUL_MINUS_ONE = -1, BLIS_MUL_ZERO, BLIS_MUL_ONE,
                 BLIS_MUL_DEFAULT };

inline const void *
bli_auxinfo_next_a(const auxinfo_t *data)
{
    return data->a_next;
}

inline const void *
bli_auxinfo_next_b(const auxinfo_t *data)
{
    return data->b_next;
}

//  C <- beta*C + Ct for an m x n tile (BLIS: bli_dxpbys_mxn)
inline void
xpbys_mxn(dim_t m, dim_t n, const double *Ct, inc_t rsCt, inc_t csCt,
          const double *beta, double *C, inc_t rsC, inc_t csC)
{
    if (*beta==0.0) {
        for (dim_t j=0; j<n; ++j) {
            for (dim_t i=0; i<m; ++i) {
                C[i*rsC+j*csC] = Ct[i*rsCt+j*csCt];
            }
        }
    } else {
        for (dim_t j=0; j<n; ++j) {
            for (dim_t i=0; i<m; ++i) {
                C[i*rsC+j*csC] = Ct[i*rsCt+j*csCt] + *beta*C[i*rsC+j*csC];
            }
        }
    }
}

//  C <- beta*C + Ct for complex tiles (BLIS: bli_cxpbys_mxn, bli_zxpbys_mxn)
template <typename T>
inline void
cxpbys_mxn(dim_t m, dim_t n, const T *Ct, inc_t rsCt, inc_t csCt,
           const T *beta, T *C, inc_t rsC, inc_t csC)
{
    const auto br = beta->real, bi = beta->imag;
    if (br==0 && bi==0) {
        for (dim_t j=0; j<n; ++j) {
            for (dim_t i=0; i<m; ++i) {
                C[i*rsC+j*csC] = Ct[i*rsCt+j*csCt];
            }
        }
    } else {
        for (dim_t j=0; j<n; ++j) {
            for (dim_t i=0; i<m; ++i) {
                T       &c  = C[i*rsC+j*csC];
                const T &ct = Ct[i*rsCt+j*csCt];
                const auto cr = c.real, ci = c.imag;
                c.real = ct.real + (br*cr - bi*ci);
                c.imag = ct.imag + (br*ci + bi*cr);
            }
        }
    }
}

inline void
xpbys_mxn(dim_t m, dim_t n, const float *Ct, inc_t rsCt, inc_t csCt,
          const float *beta, float *C, inc_t rsC, inc_t csC)
{
    if (*beta==0.0f) {
        for (dim_t j=0; j<n; ++j) {
            for (dim_t i=0; i<m; ++i) {
                C[i*rsC+j*csC] = Ct[i*rsCt+j*csCt];
            }
        }
    } else {
        for (dim_t j=0; j<n; ++j) {
            for (dim_t i=0; i<m; ++i) {
                C[i*rsC+j*csC] = Ct[i*rsCt+j*csCt] + *beta*C[i*rsC+j*csC];
            }
        }
    }
}

inline void
xpbys_mxn(dim_t m, dim_t n, const scomplex *Ct, inc_t rsCt, inc_t csCt,
          const scomplex *beta, scomplex *C, inc_t rsC, inc_t csC)
{
    cxpbys_mxn(m, n, Ct, rsCt, csCt, beta, C, rsC, csC);
}

inline void
xpbys_mxn(dim_t m, dim_t n, const dcomplex *Ct, inc_t rsCt, inc_t csCt,
          const dcomplex *beta, dcomplex *C, inc_t rsC, inc_t csC)
{
    cxpbys_mxn(m, n, Ct, rsCt, csCt, beta, C, rsC, csC);
}

} } } // namespace compat, blis, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_BLIS_COMPAT_TYPES

//
//  Macros (not guarded: defined for every kernel header, undefined by
//  compat_end.h)
//
#define restrict __restrict__

#if defined(__APPLE__)
#   define BLIS_OS_OSX 1
#else
#   define BLIS_OS_OSX 0
#endif

//  Prefix of local labels in the assembly of the kernels that use named
//  labels (see LABELS in mk/vendor_blis_kernels.py): only labels starting
//  with L are local on Mach-O, with .L on ELF.
#if defined(__APPLE__)
#   define ULMBLAS_BLIS_LABEL "L"
#else
#   define ULMBLAS_BLIS_LABEL ".L"
#endif

//  The element type of the BLIS type character 'ch' (s, d, c, z).
#define ULMBLAS_BLIS_CT_TYPE_s float
#define ULMBLAS_BLIS_CT_TYPE_d double
#define ULMBLAS_BLIS_CT_TYPE_c scomplex
#define ULMBLAS_BLIS_CT_TYPE_z dcomplex
#define ULMBLAS_BLIS_CT_TYPE(ch) ULMBLAS_BLIS_CT_TYPE_##ch

#define ULMBLAS_BLIS_SETUP_CT_PRE_(ch, mr, nr, row_major)                  \
    typedef ULMBLAS_BLIS_CT_TYPE(ch) _ctype;                               \
    const _ctype  *_beta  = static_cast<const _ctype *>(beta);             \
    _ctype        *_c     = static_cast<_ctype *>(c);                      \
    const inc_t   _rs_c   = rs_c;                                          \
    const inc_t   _cs_c   = cs_c;                                          \
    alignas(64) _ctype _ct[(mr)*(nr)];                                     \
    const inc_t   _rs_ct  = (row_major) ? (nr) : 1;                        \
    const inc_t   _cs_ct  = (row_major) ? 1 : (mr);

#define ULMBLAS_BLIS_SETUP_CT_POST_                                        \
    _ctype _zero = _ctype();                                               \
    if (_use_ct) {                                                         \
        c    = _ct;                                                        \
        rs_c = _rs_ct;                                                     \
        cs_c = _cs_ct;                                                     \
        beta = &_zero;                                                     \
    }

//  Scenario 1: the assembly supports only the preferred storage of C.
#define GEMM_UKR_SETUP_CT(ch, mr, nr, row_major)                           \
    ULMBLAS_BLIS_SETUP_CT_PRE_(ch, mr, nr, row_major)                      \
    const bool _use_ct = ((row_major) ? cs_c!=1 : rs_c!=1)                 \
                      || m!=(mr) || n!=(nr);                               \
    ULMBLAS_BLIS_SETUP_CT_POST_

//  Scenario 2: row and column storage, not general strides.
#define GEMM_UKR_SETUP_CT_AMBI(ch, mr, nr, row_major)                      \
    ULMBLAS_BLIS_SETUP_CT_PRE_(ch, mr, nr, row_major)                      \
    const bool _use_ct = (cs_c!=1 && rs_c!=1) || m!=(mr) || n!=(nr);       \
    ULMBLAS_BLIS_SETUP_CT_POST_

//  Scenario 3: any storage of C.
#define GEMM_UKR_SETUP_CT_ANY(ch, mr, nr, row_major)                       \
    ULMBLAS_BLIS_SETUP_CT_PRE_(ch, mr, nr, row_major)                      \
    const bool _use_ct = m!=(mr) || n!=(nr);                               \
    ULMBLAS_BLIS_SETUP_CT_POST_

//  Scenario 4: as 1, and C (and its stride) must be aligned.
#define GEMM_UKR_SETUP_CT_ALIGNED(ch, mr, nr, row_major, alignment)        \
    ULMBLAS_BLIS_SETUP_CT_PRE_(ch, mr, nr, row_major)                      \
    const bool _use_ct = ((row_major) ? cs_c!=1 : rs_c!=1)                 \
                      || m!=(mr) || n!=(nr)                                \
                      || (reinterpret_cast<std::uintptr_t>(_c)%(alignment))\
                      || ((((row_major) ? _rs_c : _cs_c)*sizeof(_ctype))   \
                          %(alignment));                                   \
    ULMBLAS_BLIS_SETUP_CT_POST_

#define GEMM_UKR_FLUSH_CT(ch)                                              \
    if (_use_ct) {                                                         \
        xpbys_mxn(m, n, _ct, _rs_ct, _cs_ct, _beta, _c, _rs_c, _cs_c);     \
    }

//  AOCL-BLAS tracing (aocl_dtl/aocldtl.h without tracing): no code
#define AOCL_DTL_TRACE_ENTRY(LogLevel)
#define AOCL_DTL_TRACE_EXIT(LogLevel)
#define AOCL_DTL_LEVEL_TRACE_7 12
