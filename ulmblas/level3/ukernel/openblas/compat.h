//
//  ulmBLAS: the definitions of OpenBLAS's common.h that the vendored
//  OpenBLAS kernels in kernel/*.h use (they are compiled as C++ inside
//  ulmBLAS, without common.h).  Not part of OpenBLAS.
//
#ifndef ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_COMPAT_H
#define ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_COMPAT_H 1

namespace ulmBLAS { namespace openblas { namespace compat {

//  OpenBLAS: BLASLONG is long on LP64 systems (x86-64 Linux, macOS)
typedef long BLASLONG;

} } } // namespace compat, openblas, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_COMPAT_H
