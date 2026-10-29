#ifndef ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_C_H
#define ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_C_H 1

//
//  Selection of a vendored CGEMM micro kernel (complex float) at build
//  time, independent of the DGEMM, SGEMM and ZGEMM selections.  Defining
//  exactly one macro
//
//      ULMBLAS_CGEMM_UKERNEL_blis_NAME        BLIS or AOCL-BLAS kernel
//      ULMBLAS_CGEMM_UKERNEL_openblas_NAME    OpenBLAS kernel
//
//  replaces the micro kernel for std::complex<float> (all other types keep
//  their kernels) and defines USE_BLIS_CGEMM.  The kernels are listed in
//  README.md.  The compiler flags must enable the instruction set of the
//  kernel; the CPU must support it at run time.
//
//  The macros ULMBLAS_BLIS_CGEMM_... have the meaning of the macros
//  ULMBLAS_BLIS_ZGEMM_... in select_z.h (MR and NR in complex elements).
//
//  -DULMBLAS_CGEMM_MC=..., KC, NC override the default cache block sizes
//  (MC a multiple of MR and NR, NC a multiple of NR).
//
#if defined(ULMBLAS_CGEMM_UKERNEL_blis_haswell_3x8)
#   define ULMBLAS_BLIS_CGEMM_NAME      c_haswell_3x8
#   define ULMBLAS_BLIS_CGEMM_FUNCTION  bli_cgemm_haswell_asm_3x8
#   define ULMBLAS_BLIS_CGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/c_haswell_3x8.h>
#   define ULMBLAS_BLIS_CGEMM_MR        3
#   define ULMBLAS_BLIS_CGEMM_NR        8
#   define ULMBLAS_BLIS_CGEMM_MC_       72      // BLIS haswell: 75 (not a multiple of 8), 256, 4080
#   define ULMBLAS_BLIS_CGEMM_KC_       256
#   define ULMBLAS_BLIS_CGEMM_NC_       4080
#   define ULMBLAS_BLIS_CGEMM_ALIGN     32
#   define ULMBLAS_BLIS_CGEMM_API       1
#   define ULMBLAS_BLIS_CGEMM_STORAGE   3       // GEMM_UKR_SETUP_CT, row major
#   define ULMBLAS_BLIS_CGEMM_ROWPREF   1
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_CGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_CGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_CGEMM_UKERNEL_blis_haswell_8x3)
#   define ULMBLAS_BLIS_CGEMM_NAME      c_haswell_8x3
#   define ULMBLAS_BLIS_CGEMM_FUNCTION  bli_cgemm_haswell_asm_8x3
#   define ULMBLAS_BLIS_CGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/c_haswell_8x3.h>
#   define ULMBLAS_BLIS_CGEMM_MR        8
#   define ULMBLAS_BLIS_CGEMM_NR        3
#   define ULMBLAS_BLIS_CGEMM_MC_       48      // BLIS haswell (8 x 6): 56 (not a multiple of 3), 256, 4080
#   define ULMBLAS_BLIS_CGEMM_KC_       256
#   define ULMBLAS_BLIS_CGEMM_NC_       4080
#   define ULMBLAS_BLIS_CGEMM_ALIGN     32
#   define ULMBLAS_BLIS_CGEMM_API       1
#   define ULMBLAS_BLIS_CGEMM_STORAGE   0       // GEMM_UKR_SETUP_CT, column major
#   define ULMBLAS_BLIS_CGEMM_ROWPREF   0
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_CGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_CGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_CGEMM_UKERNEL_blis_sandybridge_8x4)
#   define ULMBLAS_BLIS_CGEMM_NAME      c_sandybridge_8x4
#   define ULMBLAS_BLIS_CGEMM_FUNCTION  bli_cgemm_sandybridge_asm_8x4
#   define ULMBLAS_BLIS_CGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/c_sandybridge_8x4.h>
#   define ULMBLAS_BLIS_CGEMM_MR        8
#   define ULMBLAS_BLIS_CGEMM_NR        4
#   define ULMBLAS_BLIS_CGEMM_MC_       96      // BLIS sandybridge: 96, 256, 4096
#   define ULMBLAS_BLIS_CGEMM_KC_       256
#   define ULMBLAS_BLIS_CGEMM_NC_       4096
#   define ULMBLAS_BLIS_CGEMM_ALIGN     32
#   define ULMBLAS_BLIS_CGEMM_API       1
#   define ULMBLAS_BLIS_CGEMM_STORAGE   0       // GEMM_UKR_SETUP_CT, column major
#   define ULMBLAS_BLIS_CGEMM_ROWPREF   0
#   if !defined(__AVX__)
#       define ULMBLAS_BLIS_CGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_CGEMM_ISA       "AVX (e.g. -mavx)"
#elif defined(ULMBLAS_CGEMM_UKERNEL_blis_aocl_haswell_3x8)
//  AOCL-BLAS's version of bli_cgemm_haswell_asm_3x8 (BLIS 0.x interface),
//  used by its zen3 and zen4 configurations; block sizes of zen4 (and
//  zen3).
#   define ULMBLAS_BLIS_CGEMM_NAME      c_aocl_haswell_3x8
#   define ULMBLAS_BLIS_CGEMM_FUNCTION  bli_cgemm_haswell_asm_3x8
#   define ULMBLAS_BLIS_CGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/c_aocl_haswell_3x8.h>
#   define ULMBLAS_BLIS_CGEMM_MR        3
#   define ULMBLAS_BLIS_CGEMM_NR        8
#   define ULMBLAS_BLIS_CGEMM_MC_       144     // AOCL 5.1 zen4, zen3: 144, 256, 4080
#   define ULMBLAS_BLIS_CGEMM_KC_       256
#   define ULMBLAS_BLIS_CGEMM_NC_       4080
#   define ULMBLAS_BLIS_CGEMM_ALIGN     32
#   define ULMBLAS_BLIS_CGEMM_API       0
#   define ULMBLAS_BLIS_CGEMM_STORAGE   2       // row major, general
#   define ULMBLAS_BLIS_CGEMM_ROWPREF   1
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_CGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_CGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_CGEMM_UKERNEL_openblas_haswell_8x2)
//  OpenBLAS 0.3.34 HASWELL (cgemm_kernel_8x2_haswell.c): a kernel for a
//  whole packed block, B in panels of 2 columns.
#   define ULMBLAS_BLIS_CGEMM_NAME      c_haswell_8x2
#   define ULMBLAS_BLIS_CGEMM_NAMESPACE openblas::kernel::c_haswell_8x2
#   define ULMBLAS_BLIS_CGEMM_FUNCTION  cgemm_kernel_haswell
#   define ULMBLAS_BLIS_CGEMM_HEADER    <ulmblas/level3/ukernel/openblas/kernel/c_haswell_8x2.h>
#   define ULMBLAS_BLIS_CGEMM_MR        8
#   define ULMBLAS_BLIS_CGEMM_NR        2
//  OpenBLAS HASWELL: P 256, Q 256; R as for ZGEMM (select_z.h)
#   define ULMBLAS_BLIS_CGEMM_MC_       256
#   define ULMBLAS_BLIS_CGEMM_KC_       256
#   define ULMBLAS_BLIS_CGEMM_NC_       4096
#   define ULMBLAS_BLIS_CGEMM_ALIGN     32
#   define ULMBLAS_BLIS_CGEMM_API       2
#   define ULMBLAS_BLIS_CGEMM_STORAGE   0
#   define ULMBLAS_BLIS_CGEMM_ROWPREF   0
#   define ULMBLAS_BLIS_CGEMM_BLOCKCALL 1
#   if !(defined(__AVX__) && defined(__AVX2__) && defined(__FMA__))
#       define ULMBLAS_BLIS_CGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_CGEMM_ISA       "AVX2 and FMA (e.g. -mavx2 -mfma)"
#elif defined(ULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2)
//  OpenBLAS 0.3.34 SKYLAKEX (cgemm_kernel_8x2_skylakex.c): a kernel for a
//  whole packed block, B in panels of 2 columns.
#   define ULMBLAS_BLIS_CGEMM_NAME      c_skylakex_8x2
#   define ULMBLAS_BLIS_CGEMM_NAMESPACE openblas::kernel::c_skylakex_8x2
#   define ULMBLAS_BLIS_CGEMM_FUNCTION  cgemm_kernel_skylakex
#   define ULMBLAS_BLIS_CGEMM_HEADER    <ulmblas/level3/ukernel/openblas/kernel/c_skylakex_8x2.h>
#   define ULMBLAS_BLIS_CGEMM_MR        8
#   define ULMBLAS_BLIS_CGEMM_NR        2
//  OpenBLAS SKYLAKEX: P 384, Q 192; R as for HASWELL
#   define ULMBLAS_BLIS_CGEMM_MC_       384
#   define ULMBLAS_BLIS_CGEMM_KC_       192
#   define ULMBLAS_BLIS_CGEMM_NC_       4096
#   define ULMBLAS_BLIS_CGEMM_ALIGN     64
#   define ULMBLAS_BLIS_CGEMM_API       2
#   define ULMBLAS_BLIS_CGEMM_STORAGE   0
#   define ULMBLAS_BLIS_CGEMM_ROWPREF   0
#   define ULMBLAS_BLIS_CGEMM_BLOCKCALL 1
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_CGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_CGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#endif

#if defined(ULMBLAS_BLIS_CGEMM_NAME)

#   define USE_BLIS_CGEMM
#   define USE_BLIS_XGEMM

#   if (defined(ULMBLAS_CGEMM_UKERNEL_blis_haswell_3x8)                     \
      + defined(ULMBLAS_CGEMM_UKERNEL_blis_haswell_8x3)                     \
      + defined(ULMBLAS_CGEMM_UKERNEL_blis_sandybridge_8x4)                 \
      + defined(ULMBLAS_CGEMM_UKERNEL_blis_aocl_haswell_3x8)                \
      + defined(ULMBLAS_CGEMM_UKERNEL_openblas_haswell_8x2)                 \
      + defined(ULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2)) != 1
#       error "more than one ULMBLAS_CGEMM_UKERNEL_... defined"
#   endif

#   if !defined(__x86_64__) || defined(__ILP32__)
#       error "the vendored CGEMM micro kernels need x86-64 (LP64)"
#   endif

#   if defined(ULMBLAS_BLIS_CGEMM_ISA_MISSING)
#       error "the selected CGEMM micro kernel needs " ULMBLAS_BLIS_CGEMM_ISA
#   endif

#   if !defined(ULMBLAS_BLIS_CGEMM_NAMESPACE)
#       define ULMBLAS_BLIS_CGEMM_NAMESPACE blis::kernel::ULMBLAS_BLIS_CGEMM_NAME
#   endif
#   if !defined(ULMBLAS_BLIS_CGEMM_BLOCKCALL)
#       define ULMBLAS_BLIS_CGEMM_BLOCKCALL 0
#   endif

#   if defined(ULMBLAS_CGEMM_MC)
#       define ULMBLAS_BLIS_CGEMM_MC    ULMBLAS_CGEMM_MC
#   else
#       define ULMBLAS_BLIS_CGEMM_MC    ULMBLAS_BLIS_CGEMM_MC_
#   endif
#   if defined(ULMBLAS_CGEMM_KC)
#       define ULMBLAS_BLIS_CGEMM_KC    ULMBLAS_CGEMM_KC
#   else
#       define ULMBLAS_BLIS_CGEMM_KC    ULMBLAS_BLIS_CGEMM_KC_
#   endif
#   if defined(ULMBLAS_CGEMM_NC)
#       define ULMBLAS_BLIS_CGEMM_NC    ULMBLAS_CGEMM_NC
#   else
#       define ULMBLAS_BLIS_CGEMM_NC    ULMBLAS_BLIS_CGEMM_NC_
#   endif

#elif defined(ULMBLAS_CGEMM_MC) || defined(ULMBLAS_CGEMM_KC) \
   || defined(ULMBLAS_CGEMM_NC)
#   error "ULMBLAS_CGEMM_MC, KC, NC need a ULMBLAS_CGEMM_UKERNEL_... kernel"
#endif

#endif // ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_C_H
