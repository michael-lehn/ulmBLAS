#ifndef ULMBLAS_LEVEL3_FPSRSCOPE_H
#define ULMBLAS_LEVEL3_FPSRSCOPE_H 1

#include <ulmblas/config/simd.h>

//
//  FpsrScope: at the beginning of each level 3 driver.  With the SME
//  kernels it restores the floating-point status register once at the end
//  of the call if streaming routines ran (see
//  ulmblas/level3/ukernel/sme/fpsr.h); otherwise it does nothing.
//
#if defined(USE_SME)

#include <ulmblas/level3/ukernel/sme/fpsr.h>

namespace ulmBLAS {

struct FpsrScope
{
    FpsrScope() : restore(false) {}

    sme::FpsrRestore restore;
};

} // namespace ulmBLAS

#else

namespace ulmBLAS {

struct FpsrScope
{
    FpsrScope() {}
};

} // namespace ulmBLAS

#endif

#endif // ULMBLAS_LEVEL3_FPSRSCOPE_H
