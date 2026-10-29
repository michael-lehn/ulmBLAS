/*
   ulmBLAS: the definitions of OpenBLAS's common headers (common.h,
   common_x86_64.h of OpenBLAS 0.3.34) that the vendored OpenBLAS assembly
   kernels in this directory use, for double precision on x86-64 (System V
   ABI), without TRMMKERNEL.  Not part of OpenBLAS; the values and the
   function prologue and epilogue are those of OpenBLAS.  Only for the
   preprocessor of the assembler (*.S).

   ULMBLAS_OPENBLAS_KERNEL must be defined as the name of the function
   (OpenBLAS: CNAME); on Mach-O it gets the leading underscore of C
   symbols.
*/
#ifndef ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_ASM_COMPAT_H
#define ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_ASM_COMPAT_H 1

#if !defined(__x86_64__) || defined(__ILP32__)
#   error "the OpenBLAS assembly kernels need x86-64 (LP64)"
#endif
#if defined(_WIN32)
#   error "the OpenBLAS assembly kernels are built for the System V ABI only"
#endif
#if !defined(ULMBLAS_OPENBLAS_KERNEL)
#   error "ULMBLAS_OPENBLAS_KERNEL (the function name) is not defined"
#endif

/*  common.h: double precision */
#define DOUBLE
#define SIZE        8
#define BASE_SHIFT  3

/*  common.h: Intel CET (endbr64 at the function entry if enabled) */
#if defined(__CET__) && defined(__has_include)
#   if __has_include(<cet.h>)
#       include <cet.h>
#   endif
#endif
#ifndef _CET_ENDBR
#   define _CET_ENDBR
#endif

#define PROFCODE

#if defined(__APPLE__)
/*  common_x86_64.h, OS_DARWIN */
#   define ULMBLAS_OPENBLAS_CAT_(a, b) a ## b
#   define ULMBLAS_OPENBLAS_CAT(a, b)  ULMBLAS_OPENBLAS_CAT_(a, b)
#   define REALNAME    ULMBLAS_OPENBLAS_CAT(_, ULMBLAS_OPENBLAS_KERNEL)
#   define PROLOGUE    .text;.align 5; .globl REALNAME; REALNAME:
#   define EPILOGUE    .subsections_via_symbols
#   define ALIGN_4     .align 4
#   define ALIGN_5     .align 5
#elif defined(__ELF__)
/*  common_x86_64.h, OS_LINUX and other ELF systems */
#   define REALNAME    ULMBLAS_OPENBLAS_KERNEL
#   define PROLOGUE \
        .text; \
        .align 512; \
        .globl REALNAME ;\
       .type REALNAME, @function; \
REALNAME: \
        _CET_ENDBR
#   define EPILOGUE \
        .size    REALNAME, .-REALNAME; \
        .section .note.GNU-stack,"",@progbits
#   define ALIGN_4     .align 16
#   define ALIGN_5     .align 32
#else
#   error "the OpenBLAS assembly kernels need ELF or Mach-O"
#endif

#endif /* ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_ASM_COMPAT_H */
