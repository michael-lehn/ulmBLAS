#ifndef ULMBLAS_CONFIG_BLOCKSIZE_H
#define ULMBLAS_CONFIG_BLOCKSIZE_H 1

#include <complex>
#include <type_traits>

#include <ulmblas/config/simd.h>

namespace ulmBLAS {

#if defined(USE_TESTPARAM)

template <typename T>
struct BlockSize
{
    static constexpr int MC = USE_TESTPARAM_MC;
    static constexpr int KC = USE_TESTPARAM_KC;
    static constexpr int NC = USE_TESTPARAM_NC;


    static constexpr int MR = USE_TESTPARAM_MR;
    static constexpr int NR = USE_TESTPARAM_NR;

    //  gepack_A packs PACK_A_PANELS panels of A together if the columns of A
    //  are at least PACK_A_GROUP_DISTANCE bytes apart (not measured for this
    //  configuration).
    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    //  The packing buffers hold MC x KC (A), KC x NC (B) or MC x MC and
    //  MC x NC elements (plus MR or NR for reading ahead), panels are zero
    //  padded to multiples of MR (A) and NR (B).
    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#elif defined(USE_AVX)

//
//  AVX/FMA micro kernel for double (MR = 6, NR = 8).  MC, KC and NC for double
//  were measured on an Intel Core i5-12500 (L2 1.25 MiB per core): MC = 288
//  (an MC x KC and an MC x MC block of A fit into L2) is up to 11% faster
//  than MC = 72 for syrk, symm, trmm and trsm, and up to 3% for gemm.
//

template <typename T>
struct BlockSize
{
    static constexpr int MC
        = std::is_same<float, T>::value                ? 768
        : std::is_same<double, T>::value               ? 288
        : std::is_same<std::complex<float>, T>::value  ? 384
        : std::is_same<std::complex<double>, T>::value ? 192
        : -1;

    static constexpr int KC
        = std::is_same<float, T>::value                ? 384
        : std::is_same<double, T>::value               ? 256
        : std::is_same<std::complex<float>, T>::value  ? 384
        : std::is_same<std::complex<double>, T>::value ? 192
        : -1;

    static constexpr int NC
        = std::is_same<float, T>::value                ? 4096
        : std::is_same<double, T>::value               ? 4080
        : std::is_same<std::complex<float>, T>::value  ? 4096
        : std::is_same<std::complex<double>, T>::value ? 4096
        : -1;


    static constexpr int MR
        = std::is_same<float, T>::value                ? 8
        : std::is_same<double, T>::value               ? 6
        : std::is_same<std::complex<float>, T>::value  ? 4
        : std::is_same<std::complex<double>, T>::value ? 2
        : -1;

    static constexpr int NR
        = std::is_same<float, T>::value                ? 4
        : std::is_same<double, T>::value               ? 8
        : std::is_same<std::complex<float>, T>::value  ? 2
        : std::is_same<std::complex<double>, T>::value ? 2
        : -1;

    //  gepack_A packs PACK_A_PANELS panels of A together if the columns of A
    //  are at least PACK_A_GROUP_DISTANCE bytes apart (measured for double on
    //  an Intel Core i5-12500).
    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    //  The packing buffers hold MC x KC (A), KC x NC (B) or MC x MC and
    //  MC x NC elements (plus MR or NR for reading ahead), panels are zero
    //  padded to multiples of MR (A) and NR (B).
    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#elif defined(USE_SME)

//
//  Experimental SME2 micro kernel for double (MR = 32, NR = 16).  MC, KC
//  and NC for double were measured on an Apple M5 (MC 128 to 1024, KC 512
//  to 4096, NC 1024 to 8192; L2 16 MiB shared).  They favor small m:
//  compared with the block sizes of BLIS pull request #921 (MC = 608,
//  KC = 2240, NC = 3072) the library is 19% faster for 64 x 2048 x 2048,
//  43% for 16 x 4096 x 4096, 5% for 256 x 2048 x 2048 (the packed block of
//  B fits into L2), the same for n = 512 and 1024, but 3 to 5% slower for
//  m = n = k = 2048 to 4096, where KC = 2240 is better.  No block sizes
//  measured were best for both (campaign 004-2026-10-25-macbook-m5-sme).  The
//  packed block of A must fit well into L2: MC*KC*8 of 12 to 32 MiB was 14
//  to 43% slower for n = 4096.
//  Experimental SME2 micro kernel for complex double (MR = NR = 16, written
//  for ulmBLAS): MC = 384, KC = 1024, NC = 1024, the element counts of
//  double (KC = 1024 instead of 512 about 3% faster; MC 96 to 384 and NC
//  256 to 2048 within the noise on an Apple M5).
//  SME2 micro kernels for float (MR = NR = 32) and complex float (MR = 32,
//  NR = 16), written for ulmBLAS.  Block sizes from a sweep on an Apple M5
//  (one run per configuration under the thermal guard, square n = 256 ...
//  4000 and 64 x 2048 x 2048): float MC = 512, KC = 2048, NC = 1024 (KC =
//  2048 about 3 to 5% faster than 1024 for n >= 1000; with NC = 2048 the
//  packed block of B fills the 16 MiB L2 and 64 x 2048 x 2048 was 22%
//  slower); complex float 384, 1024, 1024 (KC = 2048 at most 2% faster for
//  large n, 9% slower for small m).
//

template <typename T>
struct BlockSize
{
    static constexpr int MC
        = std::is_same<float, T>::value                ? 512
        : std::is_same<double, T>::value               ? 384
        : std::is_same<std::complex<float>, T>::value  ? 384
        : std::is_same<std::complex<double>, T>::value ? 384
        : -1;

    static constexpr int KC
        = std::is_same<float, T>::value                ? 2048
        : std::is_same<double, T>::value               ? 1024
        : std::is_same<std::complex<float>, T>::value  ? 1024
        : std::is_same<std::complex<double>, T>::value ? 1024
        : -1;

    static constexpr int NC
        = std::is_same<float, T>::value                ? 1024
        : std::is_same<double, T>::value               ? 1024
        : std::is_same<std::complex<float>, T>::value  ? 1024
        : std::is_same<std::complex<double>, T>::value ? 1024
        : -1;


    static constexpr int MR
        = std::is_same<float, T>::value                ? 32
        : std::is_same<double, T>::value               ? 32
        : std::is_same<std::complex<float>, T>::value  ? 32
        : std::is_same<std::complex<double>, T>::value ? 16
        : -1;

    static constexpr int NR
        = std::is_same<float, T>::value                ? 32
        : std::is_same<double, T>::value               ? 16
        : std::is_same<std::complex<float>, T>::value  ? 16
        : std::is_same<std::complex<double>, T>::value ? 16
        : -1;

    //  gepack_A packs PACK_A_PANELS panels of A together if the columns of A
    //  are at least PACK_A_GROUP_DISTANCE bytes apart.  The SME packing
    //  routine for double packs panel by panel with prefetching (see
    //  ulmblas/level3/ukernel/sme/pack.tcc); grouping 2, 4, 8 or 19 panels
    //  made no measurable difference with the reference routine.
    static constexpr int PACK_A_PANELS         = 1;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    //  The packing buffers hold MC x KC (A), KC x NC (B) or MC x MC and
    //  MC x NC elements (plus MR or NR for reading ahead), panels are zero
    //  padded to multiples of MR (A) and NR (B).
    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#elif defined(USE_NEON)

//
//  AArch64 NEON micro kernel for double (MR = 6, NR = 8).  MC, KC and NC for
//  double were measured on an Apple M5 (performance cores: L1d 128 KiB, L2
//  16 MiB shared): KC = 1024 instead of 256 is 1 to 1.5% faster for dgemm
//  (fewer updates of C per element), MC = 960 instead of 240 2 to 4% faster
//  for k = 32 and for dsyrk and dtrsm (see benchmark_results).
//  NEON micro kernel for complex double (MR = 4, NR = 3, written for
//  ulmBLAS): MC = 192, KC = 512 (B panel 24 KiB, A panel 32 KiB in L1).
//  NEON micro kernels for float (MR = 8, NR = 12) and complex float (MR =
//  4, NR = 6), written for ulmBLAS.  Block sizes from a sweep on an Apple M5
//  (MC 240 ... 960 and 96 ... 384, KC 256 ... 1024): all within about 3%;
//  float MC = 960, KC = 1024, complex float MC = 384, KC = 1024.
//

template <typename T>
struct BlockSize
{
    static constexpr int MC
        = std::is_same<float, T>::value                ? 960
        : std::is_same<double, T>::value               ? 960
        : std::is_same<std::complex<float>, T>::value  ? 384
        : std::is_same<std::complex<double>, T>::value ? 192
        : -1;

    static constexpr int KC
        = std::is_same<float, T>::value                ? 1024
        : std::is_same<double, T>::value               ? 1024
        : std::is_same<std::complex<float>, T>::value  ? 1024
        : std::is_same<std::complex<double>, T>::value ? 512
        : -1;

    static constexpr int NC
        = std::is_same<float, T>::value                ? 4080
        : std::is_same<double, T>::value               ? 4080
        : std::is_same<std::complex<float>, T>::value  ? 4080
        : std::is_same<std::complex<double>, T>::value ? 4080
        : -1;


    static constexpr int MR
        = std::is_same<float, T>::value                ? 8
        : std::is_same<double, T>::value               ? 6
        : std::is_same<std::complex<float>, T>::value  ? 4
        : std::is_same<std::complex<double>, T>::value ? 4
        : -1;

    static constexpr int NR
        = std::is_same<float, T>::value                ? 12
        : std::is_same<double, T>::value               ? 8
        : std::is_same<std::complex<float>, T>::value  ? 6
        : std::is_same<std::complex<double>, T>::value ? 3
        : -1;

    //  gepack_A packs PACK_A_PANELS panels of A together if the columns of A
    //  are at least PACK_A_GROUP_DISTANCE bytes apart (measured for double on
    //  an Apple M5).
    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    //  The packing buffers hold MC x KC (A), KC x NC (B) or MC x MC and
    //  MC x NC elements (plus MR or NR for reading ahead), panels are zero
    //  padded to multiples of MR (A) and NR (B).
    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#elif defined(USE_SSE)

template <typename T>
struct BlockSize
{
    static constexpr int MC
        = std::is_same<float, T>::value                ? 768
        : std::is_same<double, T>::value               ? 384
        : std::is_same<std::complex<float>, T>::value  ? 384
        : std::is_same<std::complex<double>, T>::value ? 192
        : -1;

    static constexpr int KC
        = std::is_same<float, T>::value                ? 384
        : std::is_same<double, T>::value               ? 384
        : std::is_same<std::complex<float>, T>::value  ? 384
        : std::is_same<std::complex<double>, T>::value ? 192
        : -1;

    static constexpr int NC
        = std::is_same<float, T>::value                ? 4096
        : std::is_same<double, T>::value               ? 4096
        : std::is_same<std::complex<float>, T>::value  ? 4096
        : std::is_same<std::complex<double>, T>::value ? 4096
        : -1;


    static constexpr int MR
        = std::is_same<float, T>::value                ? 8
        : std::is_same<double, T>::value               ? 4
        : std::is_same<std::complex<float>, T>::value  ? 4
        : std::is_same<std::complex<double>, T>::value ? 2
        : -1;

    static constexpr int NR
        = std::is_same<float, T>::value                ? 4
        : std::is_same<double, T>::value               ? 4
        : std::is_same<std::complex<float>, T>::value  ? 2
        : std::is_same<std::complex<double>, T>::value ? 2
        : -1;

    //  gepack_A packs PACK_A_PANELS panels of A together if the columns of A
    //  are at least PACK_A_GROUP_DISTANCE bytes apart (measured for double on
    //  an Intel Core i5-12500).
    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    //  The packing buffers hold MC x KC (A), KC x NC (B) or MC x MC and
    //  MC x NC elements (plus MR or NR for reading ahead), panels are zero
    //  padded to multiples of MR (A) and NR (B).
    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#else

template <typename T>
struct BlockSize
{
    static constexpr int MC
        = std::is_same<float, T>::value                ? 768
        : std::is_same<double, T>::value               ? 384
        : std::is_same<std::complex<float>, T>::value  ? 384
        : std::is_same<std::complex<double>, T>::value ? 192
        : -1;

    static constexpr int KC
        = std::is_same<float, T>::value                ? 384
        : std::is_same<double, T>::value               ? 384
        : std::is_same<std::complex<float>, T>::value  ? 384
        : std::is_same<std::complex<double>, T>::value ? 192
        : -1;

    static constexpr int NC
        = std::is_same<float, T>::value                ? 4096
        : std::is_same<double, T>::value               ? 4096
        : std::is_same<std::complex<float>, T>::value  ? 4096
        : std::is_same<std::complex<double>, T>::value ? 4096
        : -1;


    static constexpr int MR
        = std::is_same<float, T>::value                ? 8
        : std::is_same<double, T>::value               ? 4
        : std::is_same<std::complex<float>, T>::value  ? 4
        : std::is_same<std::complex<double>, T>::value ? 2
        : -1;

    static constexpr int NR
        = std::is_same<float, T>::value                ? 4
        : std::is_same<double, T>::value               ? 4
        : std::is_same<std::complex<float>, T>::value  ? 2
        : std::is_same<std::complex<double>, T>::value ? 2
        : -1;

    //  gepack_A packs PACK_A_PANELS panels of A together if the columns of A
    //  are at least PACK_A_GROUP_DISTANCE bytes apart (not measured for this
    //  configuration).
    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    //  The packing buffers hold MC x KC (A), KC x NC (B) or MC x MC and
    //  MC x NC elements (plus MR or NR for reading ahead), panels are zero
    //  padded to multiples of MR (A) and NR (B).
    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#endif

#if defined(USE_BLIS_DGEMM) && !defined(USE_TESTPARAM)

//
//  BLIS-derived DGEMM micro kernel (ulmblas/level3/ukernel/blis/select.h):
//  MR and NR of the kernel, by default MC, KC and NC of the BLIS
//  configuration that uses it (-DULMBLAS_DGEMM_MC=... etc. override them).
//  The other data types keep the block sizes above.
//

template <>
struct BlockSize<double>
{
    static constexpr int MC = ULMBLAS_BLIS_DGEMM_MC;
    static constexpr int KC = ULMBLAS_BLIS_DGEMM_KC;
    static constexpr int NC = ULMBLAS_BLIS_DGEMM_NC;
    static constexpr int MR = ULMBLAS_BLIS_DGEMM_MR;
    static constexpr int NR = ULMBLAS_BLIS_DGEMM_NR;

    //  as for the AVX/FMA kernel (not measured for these kernels)
    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#endif

#if defined(USE_BLIS_SGEMM)

//
//  Vendored SGEMM micro kernel (ulmblas/level3/ukernel/blis/select_s.h): MR
//  and NR of the kernel; by default MC, KC and NC of the upstream
//  configuration that uses it (-DULMBLAS_SGEMM_MC=... etc. override them),
//  in test configurations with small block sizes (USE_TESTPARAM) MC, KC and
//  NC of the configuration.
//

template <>
struct BlockSize<float>
{
#if defined(USE_TESTPARAM)
    static constexpr int MC = USE_TESTPARAM_MC;
    static constexpr int KC = USE_TESTPARAM_KC;
    static constexpr int NC = USE_TESTPARAM_NC;
#else
    static constexpr int MC = ULMBLAS_BLIS_SGEMM_MC;
    static constexpr int KC = ULMBLAS_BLIS_SGEMM_KC;
    static constexpr int NC = ULMBLAS_BLIS_SGEMM_NC;
#endif
    static constexpr int MR = ULMBLAS_BLIS_SGEMM_MR;
    static constexpr int NR = ULMBLAS_BLIS_SGEMM_NR;

    //  as for double (not measured for these kernels)
    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#endif

#if defined(USE_BLIS_CGEMM)

//
//  Vendored CGEMM micro kernel (ulmblas/level3/ukernel/blis/select_c.h): MR
//  and NR of the kernel; by default MC, KC and NC of the upstream
//  configuration that uses it (-DULMBLAS_CGEMM_MC=... etc. override them),
//  in test configurations with small block sizes (USE_TESTPARAM) MC, KC and
//  NC of the configuration.
//

template <>
struct BlockSize<std::complex<float> >
{
#if defined(USE_TESTPARAM)
    static constexpr int MC = USE_TESTPARAM_MC;
    static constexpr int KC = USE_TESTPARAM_KC;
    static constexpr int NC = USE_TESTPARAM_NC;
#else
    static constexpr int MC = ULMBLAS_BLIS_CGEMM_MC;
    static constexpr int KC = ULMBLAS_BLIS_CGEMM_KC;
    static constexpr int NC = ULMBLAS_BLIS_CGEMM_NC;
#endif
    static constexpr int MR = ULMBLAS_BLIS_CGEMM_MR;
    static constexpr int NR = ULMBLAS_BLIS_CGEMM_NR;

    //  as for double (not measured for these kernels)
    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#endif

#if defined(USE_BLIS_ZGEMM)

//
//  Vendored ZGEMM micro kernel (ulmblas/level3/ukernel/blis/select_z.h): MR
//  and NR of the kernel; by default MC, KC and NC of the upstream
//  configuration that uses it (-DULMBLAS_ZGEMM_MC=... etc. override them),
//  in test configurations with small block sizes (USE_TESTPARAM) MC, KC and
//  NC of the configuration.
//

template <>
struct BlockSize<std::complex<double> >
{
#if defined(USE_TESTPARAM)
    static constexpr int MC = USE_TESTPARAM_MC;
    static constexpr int KC = USE_TESTPARAM_KC;
    static constexpr int NC = USE_TESTPARAM_NC;
#else
    static constexpr int MC = ULMBLAS_BLIS_ZGEMM_MC;
    static constexpr int KC = ULMBLAS_BLIS_ZGEMM_KC;
    static constexpr int NC = ULMBLAS_BLIS_ZGEMM_NC;
#endif
    static constexpr int MR = ULMBLAS_BLIS_ZGEMM_MR;
    static constexpr int NR = ULMBLAS_BLIS_ZGEMM_NR;

    //  as for double (not measured for these kernels)
    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0
                  && PACK_A_PANELS>0 && PACK_A_GROUP_DISTANCE>0,
                  "Invalid block size.");

    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#endif

#if defined(USE_TESTPARAM) && (defined(USE_NEON) || defined(USE_SME)) \
 && !defined(USE_BLIS_ZGEMM)

//
//  Test configurations with small block sizes and the NEON or SME2
//  kernels: the kernels for float, complex float and complex double fix MR
//  x NR (NEON 8 x 12, 4 x 6, 4 x 3; SME2 32 x 32, 32 x 16, 16 x 16); MC and
//  NC of the configuration are rounded down to multiples of the least
//  common multiple of MR and NR and of NR.
//
constexpr int
testparamRound(int x, int m)
{
    return (x/m*m>0) ? x/m*m : m;
}

template <>
struct BlockSize<float>
{
#if defined(USE_SME)
    static constexpr int MR = 32;
    static constexpr int NR = 32;
    static constexpr int MC = testparamRound(USE_TESTPARAM_MC, 32);
#else
    static constexpr int MR = 8;
    static constexpr int NR = 12;
    static constexpr int MC = testparamRound(USE_TESTPARAM_MC, 24);
#endif
    static constexpr int KC = USE_TESTPARAM_KC;
    static constexpr int NC = testparamRound(USE_TESTPARAM_NC, NR);

    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

template <>
struct BlockSize<std::complex<float> >
{
#if defined(USE_SME)
    static constexpr int MR = 32;
    static constexpr int NR = 16;
    static constexpr int MC = testparamRound(USE_TESTPARAM_MC, 32);
#else
    static constexpr int MR = 4;
    static constexpr int NR = 6;
    static constexpr int MC = testparamRound(USE_TESTPARAM_MC, 12);
#endif
    static constexpr int KC = USE_TESTPARAM_KC;
    static constexpr int NC = testparamRound(USE_TESTPARAM_NC, NR);

    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

template <>
struct BlockSize<std::complex<double> >
{
#if defined(USE_SME)
    static constexpr int MR = 16;
    static constexpr int NR = 16;
    static constexpr int MC = testparamRound(USE_TESTPARAM_MC, 16);
#else
    static constexpr int MR = 4;
    static constexpr int NR = 3;
    static constexpr int MC = testparamRound(USE_TESTPARAM_MC, 12);
#endif
    static constexpr int KC = USE_TESTPARAM_KC;
    static constexpr int NC = testparamRound(USE_TESTPARAM_NC, NR);

    static constexpr int PACK_A_PANELS         = 4;
    static constexpr int PACK_A_GROUP_DISTANCE = 8192;

    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#endif

} // namespace ulmBLAS

#endif // ULMBLAS_CONFIG_BLOCKSIZE_H 1
