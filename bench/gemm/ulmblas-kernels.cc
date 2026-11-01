//
//  ulmblas-kernels: the GEMM micro kernels and block sizes that ulmBLAS
//  selects with the compiler flags of this compilation, per data type.
//  run.sh (make bench-gemm) and mk/check_arch.sh (make check_arch) compile
//  it with the ARCHFLAGS in question (the selection happens at compile
//  time, ulmblas/config/simd.h, ulmblas/level3/ukernel/ugemm.h).  Output,
//  one line per type:
//
//      ulmBLAS double: kernel sse, MR x NR = 4 x 4, MC, KC, NC = 384, 384, 4096
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
//  The kernel for T.  The SSE kernel set implements double; for the other
//  types it uses the reference kernel (ulmblas/level3/ukernel/sse/ugemm.h).
//  Extend this function when kernels for other types are added.
//
template <typename T>
std::string
kernel()
{
    return std::is_same<T, double>::value ? STR(SELECT_UGEMM_KERNEL) : "ref";
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
    std::printf("ulmBLAS kernel set: %s\n", STR(SELECT_UGEMM_KERNEL));
    report<float>("float");
    report<double>("double");
    report<std::complex<float> >("complex float");
    report<std::complex<double> >("complex double");
    return 0;
}
