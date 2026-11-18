#ifndef ULMBLAS_LEVEL3_UKERNEL_KERNELBACKEND_H
#define ULMBLAS_LEVEL3_UKERNEL_KERNELBACKEND_H 1

//
//  The backend of gemm for a vendored micro kernel given by its descriptor
//  K (ulmblas/level3/ukernel/kernels.h): the frame algorithm
//  (ulmblas/level3/gemm/frame.h) runs with any kernel the build's
//  instruction set allows, not only with the one the build selects for its
//  default backend (DefaultBackend<T>, backend.h).  The selection tables of
//  gemm (ulmblas/level3/gemm/select.h, make tune) choose among them.
//
//  The calling conventions (K::API) and the storage of C the kernels accept
//  (K::STORAGE) are those of the build-time adapters (ulmblas/level3/
//  ukernel/blis/ugemm.h, xugemm.h):
//
//      API 0   BLIS 0.x and AOCL-BLAS: kc, alpha, A, B, beta, C, rs, cs
//      API 1   BLIS 1.x and 2.x: m, n, kc, alpha, A, B, beta, C, rs, cs
//      API 2   OpenBLAS: m, n, kc, alpha, A, B, C, ldc (C += alpha A B,
//              column major; any multiple of MR and NR: a panel or a block
//              in one call)
//
//      STORAGE 0  unit row stride of C       1  unit row or column stride
//              2  any strides                3  unit column stride
//              4  unit row stride, C and its columns aligned to 32 bytes
//
//  C with other strides, beta = NaN (kernels compute 0 * NaN as 0) and
//  other types of beta and C (mixed precision) go through a local tile.
//  Kernels with a limit K::KMAX on k are called for at most KMAX at a time.
//
//  KernelBackend<K>::rowPreference: the kernel prefers row-major C (the
//  frame then computes C^T = B^T A^T for column-major C).
//

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/inline.h>
#include <ulmblas/level3/pack/panel.h>
#include <ulmblas/level3/ukernel/blis/compat.h>
#include <ulmblas/level3/ukernel/kernels.h>

namespace ulmBLAS {

namespace kernels {

//  BLIS's types for T (complex: scomplex, dcomplex)
template <typename T> struct BlisType
{
    typedef T type;
};

template <> struct BlisType<std::complex<float>>
{
    typedef blis::compat::scomplex type;
};

template <> struct BlisType<std::complex<double>>
{
    typedef blis::compat::dcomplex type;
};

template <typename T>
inline bool
isNan(const T &x)
{
    return x != x;
}

template <typename T>
inline bool
isNan(const std::complex<T> &x)
{
    return x.real() != x.real() || x.imag() != x.imag();
}

//  whether the kernel K updates C with these strides directly
template <typename K>
inline bool
storageOk(std::ptrdiff_t incRowC, std::ptrdiff_t incColC, const void *C)
{
    if (K::API == 2) {
	return incRowC == 1;
    }
    switch (K::STORAGE) {
    case 0:
	return incRowC == 1;
    case 1:
	return incRowC == 1 || incColC == 1;
    case 3:
	return incColC == 1;
    case 4:
	return incRowC == 1 && reinterpret_cast<std::uintptr_t>(C) % 32 == 0 &&
	       (incColC * std::ptrdiff_t(sizeof(typename K::T))) % 32 == 0;
    default:
	return true;
    }
}

//
//  The OpenBLAS kernels (API 2): C <- C + alpha A B for an m x n block of
//  tiles, column major; beta applied before.
//
template <typename K, typename T>
inline void
openblasCall(std::int64_t m, std::int64_t n, std::int64_t k, const T &alpha,
             const T *A, const T *B, T *C, std::int64_t ldC)
{
    K::call(m, n, k, alpha, const_cast<T *>(A), const_cast<T *>(B), C, ldC);
}

template <typename K, typename R>
inline void
openblasCall(std::int64_t m, std::int64_t n, std::int64_t k,
             const std::complex<R> &alpha, const std::complex<R> *A,
             const std::complex<R> *B, std::complex<R> *C, std::int64_t ldC)
{
    K::call(m, n, k, alpha.real(), alpha.imag(),
            reinterpret_cast<R *>(const_cast<std::complex<R> *>(A)),
            reinterpret_cast<R *>(const_cast<std::complex<R> *>(B)),
            reinterpret_cast<R *>(C), ldC);
}

template <typename T>
inline void
scaleColumnMajor(std::ptrdiff_t m, std::ptrdiff_t n, const T &beta, T *C,
                 std::ptrdiff_t ldC)
{
    if (beta == T(0)) {
	for (std::ptrdiff_t j = 0; j < n; ++j) {
	    for (std::ptrdiff_t i = 0; i < m; ++i) {
		C[i + j * ldC] = T(0);
	    }
	}
    } else if (beta != T(1)) {
	for (std::ptrdiff_t j = 0; j < n; ++j) {
	    for (std::ptrdiff_t i = 0; i < m; ++i) {
		C[i + j * ldC] *= beta;
	    }
	}
    }
}

//  m x n (multiples of MR and NR) with an OpenBLAS kernel; k in steps of
//  at most KMAX only for one tile (m = MR, n = NR): the panels of several
//  tiles are kc MR and kc NR apart, the kernel expects k MR and k NR
template <typename K, typename T>
inline void
openblasBlock(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t kc,
              const T &alpha, const T *A, const T *B, const T &beta, T *C,
              std::ptrdiff_t ldC)
{
    scaleColumnMajor(m, n, beta, C, ldC);
    const std::ptrdiff_t kmax = K::KMAX > 0 ? K::KMAX : kc;
    for (std::ptrdiff_t l = 0; l < kc; l += kmax) {
	const std::ptrdiff_t k = kc - l < kmax ? kc - l : kmax;
	//  the panels are stored column by column (A: MR, B: NR per l)
	openblasCall<K>(m, n, k, alpha, A + l * K::MR, B + l * K::NR, C, ldC);
    }
}

//
//  One full tile with C the kernel accepts (storageOk, beta not NaN)
//
template <typename K, typename T>
inline void
tile(std::ptrdiff_t kc, const T &alpha, const T *A, const T *B, const T &beta,
     T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC, const T *nextA,
     const T *nextB, std::integral_constant<int, 0>)
{
    using namespace blis::compat;
    typedef typename BlisType<T>::type BT;
    auxinfo_t data = {nextA, nextB};
    K::call(dim_t(kc), reinterpret_cast<BT *>(const_cast<T *>(&alpha)),
            reinterpret_cast<BT *>(const_cast<T *>(A)),
            reinterpret_cast<BT *>(const_cast<T *>(B)),
            reinterpret_cast<BT *>(const_cast<T *>(&beta)),
            reinterpret_cast<BT *>(C), inc_t(incRowC), inc_t(incColC), &data,
            static_cast<cntx_t *>(nullptr));
}

template <typename K, typename T>
inline void
tile(std::ptrdiff_t kc, const T &alpha, const T *A, const T *B, const T &beta,
     T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC, const T *nextA,
     const T *nextB, std::integral_constant<int, 1>)
{
    using namespace blis::compat;
    auxinfo_t data = {nextA, nextB};
    K::call(dim_t(K::MR), dim_t(K::NR), dim_t(kc),
            static_cast<const void *>(&alpha), static_cast<const void *>(A),
            static_cast<const void *>(B), static_cast<const void *>(&beta),
            static_cast<void *>(C), inc_t(incRowC), inc_t(incColC),
            const_cast<const auxinfo_t *>(&data),
            static_cast<const cntx_t *>(nullptr));
}

template <typename K, typename T>
inline void
tile(std::ptrdiff_t kc, const T &alpha, const T *A, const T *B, const T &beta,
     T *C, std::ptrdiff_t, std::ptrdiff_t incColC, const T *, const T *,
     std::integral_constant<int, 2>)
{
    openblasBlock<K>(K::MR, K::NR, kc, alpha, A, B, beta, C, incColC);
}

//  a full tile, any C (through a local tile if the kernel does not accept
//  it)
template <typename K, typename T>
inline void
fullTile(std::ptrdiff_t kc, const T &alpha, const T *A, const T *B,
         const T &beta, T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC,
         const T *nextA, const T *nextB)
{
    typedef std::integral_constant<int, K::API> api;
    constexpr std::ptrdiff_t MR = K::MR, NR = K::NR;

    if (!isNan(beta) && storageOk<K>(incRowC, incColC, C)) {
	tile<K>(kc, alpha, A, B, beta, C, incRowC, incColC, nextA, nextB,
	        api());
	return;
    }
    //  kernels that store only row-major tiles (STORAGE 3) get one
    constexpr bool rowMajor = K::API == 0 && K::STORAGE == 3;
    constexpr std::ptrdiff_t rs = rowMajor ? NR : 1, cs = rowMajor ? 1 : MR;
    alignas(64) T C_[MR * NR];
    tile<K>(kc, alpha, A, B, T(0), C_, rs, cs, nextA, nextB, api());
    for (std::ptrdiff_t j = 0; j < NR; ++j) {
	for (std::ptrdiff_t i = 0; i < MR; ++i) {
	    T &c = C[i * incRowC + j * incColC];
	    c = beta == T(0) ? C_[i * rs + j * cs]
	                     : beta * c + C_[i * rs + j * cs];
	}
    }
}

//
//  Block and panel calls of the OpenBLAS kernels (API 2), for C of type T
//  with unit row stride and kc <= KMAX; otherwise (and for other kernels)
//  0, the macro kernel computes the tiles.
//
template <typename K, typename Tag, typename IndexType, typename T,
          typename Beta, typename TC>
inline IndexType
blockCall(Tag, IndexType, IndexType, IndexType, const T &, const T *, const T *,
          const Beta &, TC *, IndexType, IndexType)
{
    return 0;
}

template <typename K, typename IndexType, typename T>
inline IndexType
blockCall(std::true_type, IndexType mp, IndexType np, IndexType kc,
          const T &alpha, const T *A, const T *B, const T &beta, T *C,
          IndexType incRowC, IndexType incColC)
{
    if (!K::blockcall || mp == 0 || np == 0 || incRowC != IndexType(1) ||
        isNan(beta) || (K::KMAX > 0 && kc > K::KMAX)) {
	return 0;
    }
    openblasBlock<K>(mp * K::MR, np * K::NR, kc, alpha, A, B, beta, C, incColC);
    return np;
}

template <typename K, typename Tag, typename IndexType, typename T,
          typename Beta, typename TC>
inline IndexType
panelCall(Tag, IndexType, IndexType, const T &, const T *, const T *,
          const Beta &, TC *, IndexType, IndexType)
{
    return 0;
}

template <typename K, typename IndexType, typename T>
inline IndexType
panelCall(std::true_type, IndexType mp, IndexType kc, const T &alpha,
          const T *A, const T *B, const T &beta, T *C, IndexType incRowC,
          IndexType incColC)
{
    if (mp == 0 || incRowC != IndexType(1) || isNan(beta) ||
        (K::KMAX > 0 && kc > K::KMAX)) {
	return 0;
    }
    openblasBlock<K>(mp * K::MR, K::NR, kc, alpha, A, B, beta, C, incColC);
    return mp;
}

} // namespace kernels

template <typename K> struct KernelBackend
{
    //  the packed panels are blocks of MemoryPool (ulmblas/auxiliary/
    //  memorypool.tcc), read with aligned loads by the kernel
    static_assert(std::size_t(K::ALIGN) <= memoryPoolAlign,
                  "KernelBackend: the blocks of MemoryPool are not aligned "
                  "for the kernel (K::ALIGN)");

    typedef typename K::T T;
    static constexpr std::ptrdiff_t MR = K::MR;
    static constexpr std::ptrdiff_t NR = K::NR;
    static constexpr bool rowPreference = K::rowpref;
    typedef std::integral_constant<bool, K::API == 2> api2;

    static const char *
    name()
    {
	return K::name();
    }

    //  full tile, C of type T
    template <typename IndexType>
    static ULMBLAS_ALWAYS_INLINE void
    ugemm(IndexType kc, const T &alpha, const T *A, const T *B, const T &beta,
          T *C, IndexType incRowC, IndexType incColC, const T *nextA,
          const T *nextB)
    {
	kernels::fullTile<K>(kc, alpha, A, B, beta, C, incRowC, incColC, nextA,
	                     nextB);
    }

    //  partial tile (mr x nr) or other types of beta and C: through a local
    //  tile
    template <typename IndexType, typename Beta, typename TC>
    static void
    ugemm(IndexType mr, IndexType nr, IndexType kc, const T &alpha, const T *A,
          const T *B, const Beta &beta, TC *C, IndexType incRowC,
          IndexType incColC, const T *nextA, const T *nextB)
    {
	alignas(64) T C_[MR * NR];
	kernels::fullTile<K>(kc, alpha, A, B, T(0), C_, std::ptrdiff_t(1), MR,
	                     nextA, nextB);
	for (IndexType j = 0; j < nr; ++j) {
	    for (IndexType i = 0; i < mr; ++i) {
		TC &c = C[i * incRowC + j * incColC];
		c = beta == Beta(0) ? TC(C_[i + j * MR])
		                    : TC(beta * c + C_[i + j * MR]);
	    }
	}
    }

    template <typename IndexType, typename Beta, typename TC>
    static ULMBLAS_ALWAYS_INLINE void
    ugemm(IndexType kc, const T &alpha, const T *A, const T *B,
          const Beta &beta, TC *C, IndexType incRowC, IndexType incColC,
          const T *nextA, const T *nextB)
    {
	ugemm(IndexType(MR), IndexType(NR), kc, alpha, A, B, beta, C, incRowC,
	      incColC, nextA, nextB);
    }

    template <int P, typename IndexType, typename TA>
    static ULMBLAS_ALWAYS_INLINE void
    packA(IndexType k, bool conj, const TA *A, IndexType incRowA,
          IndexType incColA, T *buffer)
    {
	packPanelA<MR, P>(k, conj, A, incRowA, incColA, buffer);
    }

    template <typename IndexType, typename TA>
    static ULMBLAS_ALWAYS_INLINE bool
    packAPartial(IndexType, IndexType, bool, const TA *, IndexType, IndexType,
                 T *)
    {
	return false;
    }

    template <typename IndexType, typename TB>
    static ULMBLAS_ALWAYS_INLINE void
    packB(IndexType k, bool conj, const TB *B, IndexType incRowB,
          IndexType incColB, T *buffer)
    {
	packPanelB<NR>(k, conj, B, incRowB, incColB, buffer);
    }

    template <typename IndexType, typename TB>
    static ULMBLAS_ALWAYS_INLINE bool
    packBPartial(IndexType, IndexType, bool, const TB *, IndexType, IndexType,
                 T *)
    {
	return false;
    }

    //
    //  OpenBLAS kernels (API 2): mp x np full tiles in one call
    //  (K::blockcall), or a column of mp tiles, for kc <= KMAX; the number
    //  of columns of tiles or tiles done (0: none, the macro kernel computes
    //  them).
    //
    template <typename IndexType, typename Beta, typename TC>
    static IndexType
    block(IndexType mp, IndexType np, IndexType kc, const T &alpha, const T *A,
          const T *B, const Beta &beta, TC *C, IndexType incRowC,
          IndexType incColC)
    {
	return kernels::blockCall<K>(api2(), mp, np, kc, alpha, A, B, beta, C,
	                             incRowC, incColC);
    }

    template <typename IndexType, typename Beta, typename TC>
    static IndexType
    panel(IndexType mp, IndexType kc, const T &alpha, const T *A, const T *B,
          const Beta &beta, TC *C, IndexType incRowC, IndexType incColC)
    {
	return kernels::panelCall<K>(api2(), mp, kc, alpha, A, B, beta, C,
	                             incRowC, incColC);
    }
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_KERNELBACKEND_H
