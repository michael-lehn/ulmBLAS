//
//  ulmblas-kernels: the GEMM micro kernels and block sizes that ulmBLAS
//  selects with the compiler flags of this compilation, per data type.
//  run.sh (make bench-gemm) and mk/check_arch.sh (make check_arch) compile
//  it with the ARCHFLAGS in question (the selection happens at compile
//  time, ulmblas/config/simd.h, ulmblas/level3/ukernel/ugemm.h).  Output,
//  one line per type:
//
//      ulmBLAS double: kernel neon, MR x NR = 6 x 8, MC, KC, NC = 960, 1024, 4080
//
#include <complex>
#include <cstdio>
#include <string>
#include <type_traits>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/ugemm.h>

#define STR_(x) #x
#define STR(x)  STR_(x)

//
//  The kernel for T.  ulmBLAS's own kernel sets implement double (sse,
//  avx, neon, sme) and complex double (neon, sme); for the other types they
//  use the reference kernel (see their headers in ulmblas/level3/ukernel/).
//  A vendored
//  kernel replaces the kernel of its type: for double
//  ulmblas/level3/ukernel/blis/select.h, for the other types select_z.h
//  (blisk::Kernel<T>).  Extend this function when kernels for other types
//  are added.
//
template <typename T>
std::string
kernel()
{
    const std::string base = STR(ULMBLAS_BASE_UGEMM_KERNEL);
#if defined(USE_BLIS_XGEMM)
    if (ulmBLAS::blisk::Kernel<T>::selected) {
        return std::string(ulmBLAS::blisk::Kernel<T>::name()) + " (vendored)";
    }
#endif
    if (std::is_same<T, std::complex<double> >::value
        && (base=="neon" || base=="sme"))
    {
        return base;
    }
    if (!std::is_same<T, double>::value) {
        return "ref";
    }
#if defined(USE_BLIS_DGEMM)
    return std::string(STR(ULMBLAS_BLIS_DGEMM_NAME)) + " (vendored)";
#else
    return base;
#endif
}

template <typename T>
void
report(const char *type)
{
    std::printf("ulmBLAS %s: kernel %s, MR x NR = %d x %d, MC, KC, NC = %d, "
                "%d, %d\n", type, kernel<T>().c_str(),
                ulmBLAS::BlockSizeUGemm<T>::MR, ulmBLAS::BlockSizeUGemm<T>::NR,
                ulmBLAS::BlockSize<T>::MC, ulmBLAS::BlockSize<T>::KC,
                ulmBLAS::BlockSize<T>::NC);
}

int
main()
{
    std::printf("ulmBLAS kernel set: %s\n", STR(ULMBLAS_BASE_UGEMM_KERNEL));
#if defined(USE_BLIS_DGEMM)
    std::printf("ulmBLAS double: vendored kernel %s (%s)\n",
                STR(ULMBLAS_BLIS_DGEMM_NAME), STR(ULMBLAS_BLIS_DGEMM_FUNCTION));
#endif
    report<float>("float");
    report<double>("double");
    report<std::complex<float> >("complex float");
    report<std::complex<double> >("complex double");
    return 0;
}
