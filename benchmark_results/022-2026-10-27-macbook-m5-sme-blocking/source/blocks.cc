//  blocks.cc: prints the block sizes for double that a build of ulmBLAS
//  with the same flags uses (build-config.sh compares them with the
//  requested configuration).
#include <cstdio>
#include <ulmblas/config/simd.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/ugemm.h>

int
main()
{
#if defined(USE_SME)
    std::printf("USE_SME ");
#else
    std::printf("no USE_SME ");
#endif
    std::printf("MC %d KC %d NC %d MR %d NR %d\n",
                ulmBLAS::BlockSize<double>::MC,
                ulmBLAS::BlockSize<double>::KC,
                ulmBLAS::BlockSize<double>::NC,
                ulmBLAS::BlockSizeUGemm<double>::MR,
                ulmBLAS::BlockSizeUGemm<double>::NR);
}
