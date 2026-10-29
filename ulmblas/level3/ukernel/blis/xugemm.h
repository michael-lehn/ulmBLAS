#ifndef ULMBLAS_LEVEL3_UKERNEL_BLIS_XUGEMM_H
#define ULMBLAS_LEVEL3_UKERNEL_BLIS_XUGEMM_H 1

#include <complex>
#include <type_traits>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/blis/select_s.h>
#include <ulmblas/level3/ukernel/blis/select_c.h>
#include <ulmblas/level3/ukernel/blis/select_z.h>
//  the types of compat.h (its macros are only for the kernel headers)
#include <ulmblas/level3/ukernel/blis/compat.h>
#include <ulmblas/level3/ukernel/blis/compat_end.h>
#if defined(USE_BLIS_SGEMM)
#   include ULMBLAS_BLIS_SGEMM_HEADER
#endif
#if defined(USE_BLIS_CGEMM)
#   include ULMBLAS_BLIS_CGEMM_HEADER
#endif
#if defined(USE_BLIS_ZGEMM)
#   include ULMBLAS_BLIS_ZGEMM_HEADER
#endif

//
//  ulmBLAS micro kernel interface for the vendored micro kernels of the
//  types other than double (selected with -DULMBLAS_SGEMM_UKERNEL_...,
//  -DULMBLAS_CGEMM_UKERNEL_..., -DULMBLAS_ZGEMM_UKERNEL_..., see select_s.h,
//  select_c.h, select_z.h).  The adapter for double is ugemm.h; this one follows it for
//  the same three interfaces (ULMBLAS_BLIS_..._API):
//
//    1  BLIS 1.0 (m, n, k, void pointers): called with m = MR, n = NR; the
//       kernel handles the storage of C (GEMM_UKR_SETUP_CT*, compat.h);
//    0  BLIS 0.x (k, typed pointers, full tiles only): a local tile is used
//       if the assembly does not support the storage of C (..._STORAGE);
//    2  OpenBLAS (m, n, k, alpha, A, B, C, ldc; C <- C + alpha*A*B for
//       column-major C): the adapter applies beta first (C is not read for
//       beta = 0), other storage goes through a local tile; kernels with
//       ..._BLOCKCALL compute all full tiles of a packed block in one call
//       (ugemm_block).
//
//  For every interface the adapter computes alpha*A*B into a local tile and
//  updates C in C++ if beta is NaN (the kernels test beta == 0 with
//  vucomisd/vcomisd and je or sete, which also hold for NaN).
//
//  Kernel<T> describes the kernel selected for T; Kernel<T>::selected is
//  false for types without a vendored kernel.
//
namespace ulmBLAS { namespace blisk {

template <typename T>
struct Kernel
{
    static constexpr bool selected = false;
    static constexpr bool rowpref  = false;
    static constexpr bool blockcall = false;

    static const char *name() { return ""; }
};

#define ULMBLAS_BLIS_KSTR_(x) #x
#define ULMBLAS_BLIS_KSTR(x)  ULMBLAS_BLIS_KSTR_(x)

//
//  Pointer and scalar types of the kernel interfaces for T
//
template <typename T>
struct KernelTypes
{
    typedef T   blis_type;      // BLIS 0.x: float, double
    typedef T   real_type;      // OpenBLAS: element pointers
};

template <typename R>
struct KernelTypes<std::complex<R> >
{
    typedef typename std::conditional<std::is_same<R,float>::value,
                                      blis::compat::scomplex,
                                      blis::compat::dcomplex>::type blis_type;
    typedef R   real_type;
};

//  ULMBLAS_BLIS_KERNEL_TRAITS(T, X): Kernel<T> for the kernel selected with
//  the macros ULMBLAS_BLIS_XGEMM_...
#define ULMBLAS_BLIS_KERNEL_TRAITS(T, X)                                    \
template <>                                                                 \
struct Kernel<T>                                                            \
{                                                                           \
    static constexpr bool selected  = true;                                 \
    static constexpr int  MR        = ULMBLAS_BLIS_##X##GEMM_MR;            \
    static constexpr int  NR        = ULMBLAS_BLIS_##X##GEMM_NR;            \
    static constexpr int  API       = ULMBLAS_BLIS_##X##GEMM_API;           \
    static constexpr int  STORAGE   = ULMBLAS_BLIS_##X##GEMM_STORAGE;       \
    static constexpr bool rowpref   = ULMBLAS_BLIS_##X##GEMM_ROWPREF;       \
    static constexpr bool blockcall = ULMBLAS_BLIS_##X##GEMM_BLOCKCALL;     \
                                                                            \
    static const char *                                                     \
    name()                                                                  \
    {                                                                       \
        return ULMBLAS_BLIS_KSTR(ULMBLAS_BLIS_##X##GEMM_NAME);              \
    }                                                                       \
                                                                            \
    template <typename... Args>                                             \
    static void                                                             \
    call(Args... args)                                                      \
    {                                                                       \
        namespace K = ULMBLAS_BLIS_##X##GEMM_NAMESPACE;                     \
        K::ULMBLAS_BLIS_##X##GEMM_FUNCTION(args...);                        \
    }                                                                       \
};

#if defined(USE_BLIS_SGEMM)
ULMBLAS_BLIS_KERNEL_TRAITS(float, S)
#endif
#if defined(USE_BLIS_CGEMM)
ULMBLAS_BLIS_KERNEL_TRAITS(std::complex<float>, C)
#endif
#if defined(USE_BLIS_ZGEMM)
ULMBLAS_BLIS_KERNEL_TRAITS(std::complex<double>, Z)
#endif

#undef ULMBLAS_BLIS_KERNEL_TRAITS
#undef ULMBLAS_BLIS_KSTR
#undef ULMBLAS_BLIS_KSTR_

template <typename T>
struct BlockSizeKernel
{
    static constexpr int MR = Kernel<T>::MR;
    static constexpr int NR = Kernel<T>::NR;

    //  The block sizes MC, KC, NC (and the sizes of the packing buffers)
    //  are chosen for these MR, NR.
    static_assert(MR==BlockSize<T>::MR && NR==BlockSize<T>::NR,
                  "vendored micro kernel: MR and NR do not match BlockSize");
};

template <typename T>
inline bool
isNan(const T &x)
{
    return x!=x;
}

template <typename T>
inline bool
isNan(const std::complex<T> &x)
{
    return x.real()!=x.real() || x.imag()!=x.imag();
}

//
//  Calls of the kernel for one full tile
//
template <typename IndexType, typename T>
static void
xcall(std::integral_constant<int,1>, IndexType kc, const T &alpha,
      const T *A, const T *B, const T &beta, T *C, IndexType incRowC,
      IndexType incColC, const T *nextA, const T *nextB)
{
    using namespace blis::compat;
    auxinfo_t data = { nextA, nextB };

    Kernel<T>::call(dim_t(Kernel<T>::MR), dim_t(Kernel<T>::NR), dim_t(kc),
                    static_cast<const void *>(&alpha),
                    static_cast<const void *>(A),
                    static_cast<const void *>(B),
                    static_cast<const void *>(&beta),
                    static_cast<void *>(C), inc_t(incRowC), inc_t(incColC),
                    const_cast<const auxinfo_t *>(&data),
                    static_cast<const cntx_t *>(nullptr));
}

template <typename IndexType, typename T>
static void
xcall(std::integral_constant<int,0>, IndexType kc, const T &alpha,
      const T *A, const T *B, const T &beta, T *C, IndexType incRowC,
      IndexType incColC, const T *nextA, const T *nextB)
{
    using namespace blis::compat;
    typedef typename KernelTypes<T>::blis_type BT;
    auxinfo_t data = { nextA, nextB };

    Kernel<T>::call(dim_t(kc),
                    reinterpret_cast<BT *>(const_cast<T *>(&alpha)),
                    reinterpret_cast<BT *>(const_cast<T *>(A)),
                    reinterpret_cast<BT *>(const_cast<T *>(B)),
                    reinterpret_cast<BT *>(const_cast<T *>(&beta)),
                    reinterpret_cast<BT *>(C), inc_t(incRowC), inc_t(incColC),
                    &data, static_cast<cntx_t *>(nullptr));
}

//  OpenBLAS: C <- C + alpha*A*B for an m x n block of column-major C
template <typename T>
static void
obcall(long m, long n, long k, const T &alpha, const T *A, const T *B, T *C,
       long ldc)
{
    Kernel<T>::call(m, n, k, alpha, const_cast<T *>(A), const_cast<T *>(B),
                    C, ldc);
}

template <typename R>
static void
obcall(long m, long n, long k, const std::complex<R> &alpha,
       const std::complex<R> *A, const std::complex<R> *B,
       std::complex<R> *C, long ldc)
{
    Kernel<std::complex<R> >::call(m, n, k, alpha.real(), alpha.imag(),
                                   reinterpret_cast<R *>(
                                       const_cast<std::complex<R> *>(A)),
                                   reinterpret_cast<R *>(
                                       const_cast<std::complex<R> *>(B)),
                                   reinterpret_cast<R *>(C), ldc);
}

//  C <- beta*C for an m x n block of column-major C (C not read for beta = 0)
template <typename IndexType, typename T>
static void
obscal(IndexType m, IndexType n, const T &beta, T *C, IndexType incColC)
{
    if (beta==T(0)) {
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=0; i<m; ++i) {
                C[i+j*incColC] = T(0);
            }
        }
    } else if (beta!=T(1)) {
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=0; i<m; ++i) {
                C[i+j*incColC] *= beta;
            }
        }
    }
}

template <typename IndexType, typename T>
static void
xcall(std::integral_constant<int,2>, IndexType kc, const T &alpha,
      const T *A, const T *B, const T &beta, T *C, IndexType /*incRowC*/,
      IndexType incColC, const T *, const T *)
{
    constexpr IndexType MR = Kernel<T>::MR;
    constexpr IndexType NR = Kernel<T>::NR;

    obscal(MR, NR, beta, C, incColC);
    obcall(long(MR), long(NR), long(kc), alpha, A, B, C, long(incColC));
}

//
//  Storage of C that the assembly of the kernel supports (the others go
//  through a local tile)
//
template <int API, int STORAGE>
struct StorageOk
{
    template <typename IndexType>
    static bool
    check(IndexType incRowC, IndexType incColC)
    {
        return API==1
            || (API==2 && incRowC==IndexType(1))
            || (API==0 && STORAGE==0 && incRowC==IndexType(1))
            || (API==0 && STORAGE==1 && (incRowC==IndexType(1)
                                         || incColC==IndexType(1)))
            || (API==0 && STORAGE==2)
            || (API==0 && STORAGE==3 && incColC==IndexType(1));
    }
};

template <typename IndexType, typename T>
static typename std::enable_if<Kernel<T>::selected
                            && std::is_convertible<IndexType,long>::value,
void>::type
xugemm(IndexType   kc,
       const T     &alpha,
       const T     *A,
       const T     *B,
       const T     &beta,
       T           *C,
       IndexType   incRowC,
       IndexType   incColC,
       const T     *nextA,
       const T     *nextB)
{
    typedef Kernel<T> K;
    constexpr IndexType MR = K::MR;
    constexpr IndexType NR = K::NR;
    typedef std::integral_constant<int,K::API> api;

    const bool storageOk = StorageOk<K::API,K::STORAGE>::check(incRowC,
                                                               incColC);

    if (isNan(beta) || !storageOk) {
        //  local tile in a storage the kernel supports
        constexpr bool rowMajor = K::API==0 && K::STORAGE==3;
        constexpr IndexType rs = rowMajor ? NR : 1;
        constexpr IndexType cs = rowMajor ? 1 : MR;
        alignas(64) T C_[MR*NR];

        xcall(api(), kc, alpha, A, B, T(0), C_, rs, cs, nextA, nextB);
        if (beta==T(0)) {
            for (IndexType j=0; j<NR; ++j) {
                for (IndexType i=0; i<MR; ++i) {
                    C[i*incRowC+j*incColC] = C_[i*rs+j*cs];
                }
            }
        } else {
            for (IndexType j=0; j<NR; ++j) {
                for (IndexType i=0; i<MR; ++i) {
                    C[i*incRowC+j*incColC] = beta*C[i*incRowC+j*incColC]
                                           + C_[i*rs+j*cs];
                }
            }
        }
        return;
    }
    xcall(api(), kc, alpha, A, B, beta, C, incRowC, incColC, nextA, nextB);
}

//
//  Block variant for OpenBLAS kernels with ..._BLOCKCALL: C <- beta*C +
//  alpha*A*B for the mp x np full MR x NR tiles of a packed block (A: mp
//  panels of MR rows, B: np panels of NR columns, column-major C) with one
//  call of the kernel, as OpenBLAS's driver calls it.
//
template <typename IndexType, typename T>
static typename std::enable_if<Kernel<T>::blockcall, bool>::type
xugemm_block(IndexType   mp,
             IndexType   np,
             IndexType   kc,
             const T     &alpha,
             const T     *A,
             const T     *B,
             const T     &beta,
             T           *C,
             IndexType   incColC)
{
    const IndexType m = mp*Kernel<T>::MR;
    const IndexType n = np*Kernel<T>::NR;

    obscal(m, n, beta, C, incColC);
    obcall(long(m), long(n), long(kc), alpha, A, B, C, long(incColC));
    return true;
}

} } // namespace blisk, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_BLIS_XUGEMM_H
