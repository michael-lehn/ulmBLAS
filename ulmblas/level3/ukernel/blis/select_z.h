#ifndef ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_Z_H
#define ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_Z_H 1

//
//  Selection of a vendored ZGEMM micro kernel (complex double) at build
//  time, independent of the DGEMM selection (select.h).  Defining exactly
//  one macro
//
//      ULMBLAS_ZGEMM_UKERNEL_blis_NAME        BLIS or AOCL-BLAS kernel
//      ULMBLAS_ZGEMM_UKERNEL_openblas_NAME    OpenBLAS kernel
//
//  replaces the micro kernel for std::complex<double> (all other types keep
//  their kernels) and defines USE_BLIS_ZGEMM.  The kernels are listed in
//  README.md.  The compiler flags must enable the instruction set of the
//  kernel; the CPU must support it at run time.
//
//  The macros ULMBLAS_BLIS_ZGEMM_... have the meaning of the macros
//  ULMBLAS_BLIS_DGEMM_... in select.h (MR, NR, MC, KC, NC, ALIGN, API,
//  STORAGE, NAMESPACE, BLOCKCALL), MR and NR in complex elements, and
//
//      ..._ROWPREF      1: the kernel prefers row-major C (BLIS's
//                       BLIS_GEMM_UKR_ROW_PREF): ZGEMM computes C^T = B^T A^T
//                       if C is column major (gemm.tcc), as BLIS does
//
//  -DULMBLAS_ZGEMM_MC=..., KC, NC override the default cache block sizes
//  (MC a multiple of MR and NR, NC a multiple of NR).
//
#if defined(ULMBLAS_ZGEMM_UKERNEL_blis_haswell_3x4)
#   define ULMBLAS_BLIS_ZGEMM_NAME      z_haswell_3x4
#   define ULMBLAS_BLIS_ZGEMM_FUNCTION  bli_zgemm_haswell_asm_3x4
#   define ULMBLAS_BLIS_ZGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/z_haswell_3x4.h>
#   define ULMBLAS_BLIS_ZGEMM_MR        3
#   define ULMBLAS_BLIS_ZGEMM_NR        4
#   define ULMBLAS_BLIS_ZGEMM_MC_       192     // BLIS haswell: 192, 256, 4080
#   define ULMBLAS_BLIS_ZGEMM_KC_       256
#   define ULMBLAS_BLIS_ZGEMM_NC_       4080
#   define ULMBLAS_BLIS_ZGEMM_ALIGN     32
#   define ULMBLAS_BLIS_ZGEMM_API       1
#   define ULMBLAS_BLIS_ZGEMM_STORAGE   3       // GEMM_UKR_SETUP_CT, row major
#   define ULMBLAS_BLIS_ZGEMM_ROWPREF   1
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_ZGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_ZGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_ZGEMM_UKERNEL_blis_haswell_4x3)
#   define ULMBLAS_BLIS_ZGEMM_NAME      z_haswell_4x3
#   define ULMBLAS_BLIS_ZGEMM_FUNCTION  bli_zgemm_haswell_asm_4x3
#   define ULMBLAS_BLIS_ZGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/z_haswell_4x3.h>
#   define ULMBLAS_BLIS_ZGEMM_MR        4
#   define ULMBLAS_BLIS_ZGEMM_NR        3
#   define ULMBLAS_BLIS_ZGEMM_MC_       36      // BLIS haswell (8 x 6): 44 (not a multiple of 3), 256, 4080
#   define ULMBLAS_BLIS_ZGEMM_KC_       256
#   define ULMBLAS_BLIS_ZGEMM_NC_       4080
#   define ULMBLAS_BLIS_ZGEMM_ALIGN     32
#   define ULMBLAS_BLIS_ZGEMM_API       1
#   define ULMBLAS_BLIS_ZGEMM_STORAGE   0       // GEMM_UKR_SETUP_CT, column major
#   define ULMBLAS_BLIS_ZGEMM_ROWPREF   0
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_ZGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_ZGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_ZGEMM_UKERNEL_blis_sandybridge_4x4)
#   define ULMBLAS_BLIS_ZGEMM_NAME      z_sandybridge_4x4
#   define ULMBLAS_BLIS_ZGEMM_FUNCTION  bli_zgemm_sandybridge_asm_4x4
#   define ULMBLAS_BLIS_ZGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/z_sandybridge_4x4.h>
#   define ULMBLAS_BLIS_ZGEMM_MR        4
#   define ULMBLAS_BLIS_ZGEMM_NR        4
#   define ULMBLAS_BLIS_ZGEMM_MC_       64      // BLIS sandybridge: 64, 192, 4096
#   define ULMBLAS_BLIS_ZGEMM_KC_       192
#   define ULMBLAS_BLIS_ZGEMM_NC_       4096
#   define ULMBLAS_BLIS_ZGEMM_ALIGN     32
#   define ULMBLAS_BLIS_ZGEMM_API       1
#   define ULMBLAS_BLIS_ZGEMM_STORAGE   0       // GEMM_UKR_SETUP_CT, column major
#   define ULMBLAS_BLIS_ZGEMM_ROWPREF   0
#   if !defined(__AVX__)
#       define ULMBLAS_BLIS_ZGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_ZGEMM_ISA       "AVX (e.g. -mavx)"
#elif defined(ULMBLAS_ZGEMM_UKERNEL_blis_zen4_12x4)
#   define ULMBLAS_BLIS_ZGEMM_NAME      z_zen4_12x4
#   define ULMBLAS_BLIS_ZGEMM_FUNCTION  bli_zgemm_zen4_asm_12x4
#   define ULMBLAS_BLIS_ZGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/z_zen4_12x4.h>
#   define ULMBLAS_BLIS_ZGEMM_MR        12
#   define ULMBLAS_BLIS_ZGEMM_NR        4
#   define ULMBLAS_BLIS_ZGEMM_MC_       60      // AOCL 5.1 zen4, zen5: 60, 512, 2004
#   define ULMBLAS_BLIS_ZGEMM_KC_       512
#   define ULMBLAS_BLIS_ZGEMM_NC_       2004
#   define ULMBLAS_BLIS_ZGEMM_ALIGN     64
#   define ULMBLAS_BLIS_ZGEMM_API       0
#   define ULMBLAS_BLIS_ZGEMM_STORAGE   2       // column, row major, general
#   define ULMBLAS_BLIS_ZGEMM_ROWPREF   0
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__) \
      && defined(__AVX512DQ__))
#       define ULMBLAS_BLIS_ZGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_ZGEMM_ISA       "AVX, FMA, AVX-512F and AVX-512DQ (e.g. -mavx -mfma -mavx512f -mavx512dq)"
#elif defined(ULMBLAS_ZGEMM_UKERNEL_blis_zen4_4x12)
//  AOCL registers this kernel for ZTRSM (block sizes of TRSM in zen4).
#   define ULMBLAS_BLIS_ZGEMM_NAME      z_zen4_4x12
#   define ULMBLAS_BLIS_ZGEMM_FUNCTION  bli_zgemm_zen4_asm_4x12
#   define ULMBLAS_BLIS_ZGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/z_zen4_4x12.h>
#   define ULMBLAS_BLIS_ZGEMM_MR        4
#   define ULMBLAS_BLIS_ZGEMM_NR        12
#   define ULMBLAS_BLIS_ZGEMM_MC_       36      // AOCL 5.1 zen4, TRSM: 40 (not a multiple of 12), 512, 2004
#   define ULMBLAS_BLIS_ZGEMM_KC_       512
#   define ULMBLAS_BLIS_ZGEMM_NC_       2004
#   define ULMBLAS_BLIS_ZGEMM_ALIGN     64
#   define ULMBLAS_BLIS_ZGEMM_API       0
#   define ULMBLAS_BLIS_ZGEMM_STORAGE   2       // row major, general
#   define ULMBLAS_BLIS_ZGEMM_ROWPREF   1
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__) \
      && defined(__AVX512DQ__))
#       define ULMBLAS_BLIS_ZGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_ZGEMM_ISA       "AVX, FMA, AVX-512F and AVX-512DQ (e.g. -mavx -mfma -mavx512f -mavx512dq)"
#elif defined(ULMBLAS_ZGEMM_UKERNEL_blis_aocl_haswell_3x4)
//  AOCL-BLAS's version of bli_zgemm_haswell_asm_3x4 (BLIS 0.x interface),
//  used by its zen, zen2, zen3 configurations; block sizes of zen3.
#   define ULMBLAS_BLIS_ZGEMM_NAME      z_aocl_haswell_3x4
#   define ULMBLAS_BLIS_ZGEMM_FUNCTION  bli_zgemm_haswell_asm_3x4
#   define ULMBLAS_BLIS_ZGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/z_aocl_haswell_3x4.h>
#   define ULMBLAS_BLIS_ZGEMM_MR        3
#   define ULMBLAS_BLIS_ZGEMM_NR        4
#   define ULMBLAS_BLIS_ZGEMM_MC_       12      // AOCL 5.1 zen3: 18 (not a multiple of 4), 566, 256
#   define ULMBLAS_BLIS_ZGEMM_KC_       566
#   define ULMBLAS_BLIS_ZGEMM_NC_       256
#   define ULMBLAS_BLIS_ZGEMM_ALIGN     32
#   define ULMBLAS_BLIS_ZGEMM_API       0
#   define ULMBLAS_BLIS_ZGEMM_STORAGE   2       // row major, general
#   define ULMBLAS_BLIS_ZGEMM_ROWPREF   1
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_ZGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_ZGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_ZGEMM_UKERNEL_openblas_haswell_4x2)
//  OpenBLAS 0.3.34 HASWELL (zgemm_kernel_4x2_haswell.c): a kernel for a
//  whole packed block (4 x 6 register tiles), B in panels of 2 columns.
#   define ULMBLAS_BLIS_ZGEMM_NAME      z_haswell_4x2
#   define ULMBLAS_BLIS_ZGEMM_NAMESPACE openblas::kernel::z_haswell_4x2
#   define ULMBLAS_BLIS_ZGEMM_FUNCTION  zgemm_kernel_haswell
#   define ULMBLAS_BLIS_ZGEMM_HEADER    <ulmblas/level3/ukernel/openblas/kernel/z_haswell_4x2.h>
#   define ULMBLAS_BLIS_ZGEMM_MR        4
#   define ULMBLAS_BLIS_ZGEMM_NR        2
//  OpenBLAS HASWELL: P 192, Q 192; R = zgemm_r is computed at run time
//  from the buffer size (larger than 4096): NC = 4096 blocks n <= 4096 as
//  OpenBLAS does and bounds the packing buffer of B.
#   define ULMBLAS_BLIS_ZGEMM_MC_       192
#   define ULMBLAS_BLIS_ZGEMM_KC_       192
#   define ULMBLAS_BLIS_ZGEMM_NC_       4096
#   define ULMBLAS_BLIS_ZGEMM_ALIGN     32
#   define ULMBLAS_BLIS_ZGEMM_API       2
#   define ULMBLAS_BLIS_ZGEMM_STORAGE   0
#   define ULMBLAS_BLIS_ZGEMM_ROWPREF   0
#   define ULMBLAS_BLIS_ZGEMM_BLOCKCALL 1
#   if !(defined(__AVX__) && defined(__AVX2__) && defined(__FMA__))
#       define ULMBLAS_BLIS_ZGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_ZGEMM_ISA       "AVX2 and FMA (e.g. -mavx2 -mfma)"
#elif defined(ULMBLAS_ZGEMM_UKERNEL_openblas_skylakex_4x2)
//  OpenBLAS 0.3.34 SKYLAKEX (zgemm_kernel_4x2_skylakex.c): a kernel for a
//  whole packed block (4 x 12 register tiles), B in panels of 2 columns.
#   define ULMBLAS_BLIS_ZGEMM_NAME      z_skylakex_4x2
#   define ULMBLAS_BLIS_ZGEMM_NAMESPACE openblas::kernel::z_skylakex_4x2
#   define ULMBLAS_BLIS_ZGEMM_FUNCTION  zgemm_kernel_skylakex
#   define ULMBLAS_BLIS_ZGEMM_HEADER    <ulmblas/level3/ukernel/openblas/kernel/z_skylakex_4x2.h>
#   define ULMBLAS_BLIS_ZGEMM_MR        4
#   define ULMBLAS_BLIS_ZGEMM_NR        2
//  OpenBLAS SKYLAKEX: P 256, Q 128; R as for HASWELL
#   define ULMBLAS_BLIS_ZGEMM_MC_       256
#   define ULMBLAS_BLIS_ZGEMM_KC_       128
#   define ULMBLAS_BLIS_ZGEMM_NC_       4096
#   define ULMBLAS_BLIS_ZGEMM_ALIGN     64
#   define ULMBLAS_BLIS_ZGEMM_API       2
#   define ULMBLAS_BLIS_ZGEMM_STORAGE   0
#   define ULMBLAS_BLIS_ZGEMM_ROWPREF   0
#   define ULMBLAS_BLIS_ZGEMM_BLOCKCALL 1
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_ZGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_ZGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#endif

#if defined(ULMBLAS_BLIS_ZGEMM_NAME)

#   define USE_BLIS_ZGEMM
#   define USE_BLIS_XGEMM

#   if (defined(ULMBLAS_ZGEMM_UKERNEL_blis_haswell_3x4)                     \
      + defined(ULMBLAS_ZGEMM_UKERNEL_blis_haswell_4x3)                     \
      + defined(ULMBLAS_ZGEMM_UKERNEL_blis_sandybridge_4x4)                 \
      + defined(ULMBLAS_ZGEMM_UKERNEL_blis_zen4_12x4)                       \
      + defined(ULMBLAS_ZGEMM_UKERNEL_blis_zen4_4x12)                       \
      + defined(ULMBLAS_ZGEMM_UKERNEL_blis_aocl_haswell_3x4)                \
      + defined(ULMBLAS_ZGEMM_UKERNEL_openblas_haswell_4x2)                 \
      + defined(ULMBLAS_ZGEMM_UKERNEL_openblas_skylakex_4x2)) != 1
#       error "more than one ULMBLAS_ZGEMM_UKERNEL_... defined"
#   endif

#   if !defined(__x86_64__) || defined(__ILP32__)
#       error "the vendored ZGEMM micro kernels need x86-64 (LP64)"
#   endif

#   if defined(ULMBLAS_BLIS_ZGEMM_ISA_MISSING)
#       error "the selected ZGEMM micro kernel needs " ULMBLAS_BLIS_ZGEMM_ISA
#   endif

#   if !defined(ULMBLAS_BLIS_ZGEMM_NAMESPACE)
#       define ULMBLAS_BLIS_ZGEMM_NAMESPACE blis::kernel::ULMBLAS_BLIS_ZGEMM_NAME
#   endif
#   if !defined(ULMBLAS_BLIS_ZGEMM_BLOCKCALL)
#       define ULMBLAS_BLIS_ZGEMM_BLOCKCALL 0
#   endif

#   if defined(ULMBLAS_ZGEMM_MC)
#       define ULMBLAS_BLIS_ZGEMM_MC    ULMBLAS_ZGEMM_MC
#   else
#       define ULMBLAS_BLIS_ZGEMM_MC    ULMBLAS_BLIS_ZGEMM_MC_
#   endif
#   if defined(ULMBLAS_ZGEMM_KC)
#       define ULMBLAS_BLIS_ZGEMM_KC    ULMBLAS_ZGEMM_KC
#   else
#       define ULMBLAS_BLIS_ZGEMM_KC    ULMBLAS_BLIS_ZGEMM_KC_
#   endif
#   if defined(ULMBLAS_ZGEMM_NC)
#       define ULMBLAS_BLIS_ZGEMM_NC    ULMBLAS_ZGEMM_NC
#   else
#       define ULMBLAS_BLIS_ZGEMM_NC    ULMBLAS_BLIS_ZGEMM_NC_
#   endif

#elif defined(ULMBLAS_ZGEMM_MC) || defined(ULMBLAS_ZGEMM_KC) \
   || defined(ULMBLAS_ZGEMM_NC)
#   error "ULMBLAS_ZGEMM_MC, KC, NC need a ULMBLAS_ZGEMM_UKERNEL_... kernel"
#endif

#endif // ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_Z_H
