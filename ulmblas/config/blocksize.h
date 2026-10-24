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
    static const int MC = USE_TESTPARAM_MC;
    static const int KC = USE_TESTPARAM_KC;
    static const int NC = USE_TESTPARAM_NC;


    static const int MR = USE_TESTPARAM_MR;
    static const int NR = USE_TESTPARAM_NR;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");

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
    static const int MC = std::is_same<float, T>::value                ? 768
                        : std::is_same<double, T>::value               ? 288
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int KC = std::is_same<float, T>::value                ? 384
                        : std::is_same<double, T>::value               ? 256
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int NC = std::is_same<float, T>::value                ? 4096
                        : std::is_same<double, T>::value               ? 4080
                        : std::is_same<std::complex<float>, T>::value  ? 4096
                        : std::is_same<std::complex<double>, T>::value ? 4096
                        : -1;


    static const int MR = std::is_same<float, T>::value                ? 8
                        : std::is_same<double, T>::value               ? 6
                        : std::is_same<std::complex<float>, T>::value  ? 4
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static const int NR = std::is_same<float, T>::value                ? 4
                        : std::is_same<double, T>::value               ? 8
                        : std::is_same<std::complex<float>, T>::value  ? 2
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");

    //  The packing buffers hold MC x KC (A), KC x NC (B) or MC x MC and
    //  MC x NC elements (plus MR or NR for reading ahead), panels are zero
    //  padded to multiples of MR (A) and NR (B).
    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#elif defined(USE_NEON)

//
//  AArch64 NEON micro kernel for double (MR = 6, NR = 8).
//

template <typename T>
struct BlockSize
{
    static const int MC = std::is_same<float, T>::value                ? 768
                        : std::is_same<double, T>::value               ? 240
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int KC = std::is_same<float, T>::value                ? 384
                        : std::is_same<double, T>::value               ? 256
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int NC = std::is_same<float, T>::value                ? 4096
                        : std::is_same<double, T>::value               ? 4080
                        : std::is_same<std::complex<float>, T>::value  ? 4096
                        : std::is_same<std::complex<double>, T>::value ? 4096
                        : -1;


    static const int MR = std::is_same<float, T>::value                ? 8
                        : std::is_same<double, T>::value               ? 6
                        : std::is_same<std::complex<float>, T>::value  ? 4
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static const int NR = std::is_same<float, T>::value                ? 4
                        : std::is_same<double, T>::value               ? 8
                        : std::is_same<std::complex<float>, T>::value  ? 2
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");

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
    static const int MC = std::is_same<float, T>::value                ? 768
                        : std::is_same<double, T>::value               ? 384
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int KC = std::is_same<float, T>::value                ? 384
                        : std::is_same<double, T>::value               ? 384
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int NC = std::is_same<float, T>::value                ? 4096
                        : std::is_same<double, T>::value               ? 4096
                        : std::is_same<std::complex<float>, T>::value  ? 4096
                        : std::is_same<std::complex<double>, T>::value ? 4096
                        : -1;


    static const int MR = std::is_same<float, T>::value                ? 8
                        : std::is_same<double, T>::value               ? 4
                        : std::is_same<std::complex<float>, T>::value  ? 4
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static const int NR = std::is_same<float, T>::value                ? 4
                        : std::is_same<double, T>::value               ? 4
                        : std::is_same<std::complex<float>, T>::value  ? 2
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");

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
    static const int MC = std::is_same<float, T>::value                ? 768
                        : std::is_same<double, T>::value               ? 384
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int KC = std::is_same<float, T>::value                ? 384
                        : std::is_same<double, T>::value               ? 384
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int NC = std::is_same<float, T>::value                ? 4096
                        : std::is_same<double, T>::value               ? 4096
                        : std::is_same<std::complex<float>, T>::value  ? 4096
                        : std::is_same<std::complex<double>, T>::value ? 4096
                        : -1;


    static const int MR = std::is_same<float, T>::value                ? 8
                        : std::is_same<double, T>::value               ? 4
                        : std::is_same<std::complex<float>, T>::value  ? 4
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static const int NR = std::is_same<float, T>::value                ? 4
                        : std::is_same<double, T>::value               ? 4
                        : std::is_same<std::complex<float>, T>::value  ? 2
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");

    //  The packing buffers hold MC x KC (A), KC x NC (B) or MC x MC and
    //  MC x NC elements (plus MR or NR for reading ahead), panels are zero
    //  padded to multiples of MR (A) and NR (B).
    static_assert(MC%MR==0 && MC%NR==0 && NC%NR==0,
                  "MC must be a multiple of MR and NR, NC a multiple of NR.");
};

#endif



} // namespace ulmBLAS

#endif // ULMBLAS_CONFIG_BLOCKSIZE_H 1
