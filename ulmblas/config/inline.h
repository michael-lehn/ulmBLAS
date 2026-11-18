#ifndef ULMBLAS_CONFIG_INLINE_H
#define ULMBLAS_CONFIG_INLINE_H 1

//
//  ULMBLAS_ALWAYS_INLINE: for functions that only forward to another one
//  (e.g. the functions of a backend, ulmblas/level3/ukernel/backend.h), so
//  that an additional layer does not change what the compiler inlines.
//
#if defined(__GNUC__)
#define ULMBLAS_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define ULMBLAS_ALWAYS_INLINE __forceinline
#else
#define ULMBLAS_ALWAYS_INLINE inline
#endif

//
//  ULMBLAS_NOINLINE: for a function whose call in a loop must stay a call,
//  e.g. a division that the vectorizer must not apply to lanes beyond the
//  loop (TrsmFrame::reciprocal, ulmblas/level3/gemm/frametrsm.h).
//
#if defined(__GNUC__)
#define ULMBLAS_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define ULMBLAS_NOINLINE __declspec(noinline)
#else
#define ULMBLAS_NOINLINE
#endif

#endif // ULMBLAS_CONFIG_INLINE_H
