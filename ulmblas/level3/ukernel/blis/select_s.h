#ifndef ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_S_H
#define ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_S_H 1

//
//  Selection of a vendored SGEMM micro kernel (float) at build time,
//  independent of the DGEMM and ZGEMM selections (select.h, select_z.h).
//  Defining exactly one macro
//
//      ULMBLAS_SGEMM_UKERNEL_blis_NAME        BLIS or AOCL-BLAS kernel
//      ULMBLAS_SGEMM_UKERNEL_openblas_NAME    OpenBLAS kernel
//
//  replaces the micro kernel for float (all other types keep their
//  kernels) and defines USE_BLIS_SGEMM.  The macros ULMBLAS_BLIS_SGEMM_...
//  have the meaning of those in select_z.h.  -DULMBLAS_SGEMM_MC=..., KC, NC
//  override the default cache block sizes (MC a multiple of MR and NR, NC
//  a multiple of NR).
//
#if defined(ULMBLAS_SGEMM_UKERNEL_blis_haswell_6x16)
#   define ULMBLAS_BLIS_SGEMM_NAME      s_haswell_6x16
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  bli_sgemm_haswell_asm_6x16
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/s_haswell_6x16.h>
#   define ULMBLAS_BLIS_SGEMM_MR        6
#   define ULMBLAS_BLIS_SGEMM_NR        16
#   define ULMBLAS_BLIS_SGEMM_MC_       144     // BLIS haswell: 168 (not a multiple of 16), 256, 4080
#   define ULMBLAS_BLIS_SGEMM_KC_       256
#   define ULMBLAS_BLIS_SGEMM_NC_       4080
#   define ULMBLAS_BLIS_SGEMM_ALIGN     32
#   define ULMBLAS_BLIS_SGEMM_API       1
#   define ULMBLAS_BLIS_SGEMM_STORAGE   1       // GEMM_UKR_SETUP_CT_AMBI
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   1
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_SGEMM_UKERNEL_blis_haswell_16x6)
#   define ULMBLAS_BLIS_SGEMM_NAME      s_haswell_16x6
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  bli_sgemm_haswell_asm_16x6
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/s_haswell_16x6.h>
#   define ULMBLAS_BLIS_SGEMM_MR        16
#   define ULMBLAS_BLIS_SGEMM_NR        6
#   define ULMBLAS_BLIS_SGEMM_MC_       96      // BLIS haswell (8 x 6 family): 112 (not a multiple of 6), 256, 4080
#   define ULMBLAS_BLIS_SGEMM_KC_       256
#   define ULMBLAS_BLIS_SGEMM_NC_       4080
#   define ULMBLAS_BLIS_SGEMM_ALIGN     32
#   define ULMBLAS_BLIS_SGEMM_API       1
#   define ULMBLAS_BLIS_SGEMM_STORAGE   0       // GEMM_UKR_SETUP_CT, column major
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   0
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_SGEMM_UKERNEL_blis_haswell_24x4)
//  BLIS 0.x kernel from kernels/haswell/3/old/ (no BLIS configuration uses
//  it): block sizes of the haswell configuration.
#   define ULMBLAS_BLIS_SGEMM_NAME      s_haswell_24x4
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  bli_sgemm_haswell_asm_24x4
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/s_haswell_24x4.h>
#   define ULMBLAS_BLIS_SGEMM_MR        24
#   define ULMBLAS_BLIS_SGEMM_NR        4
#   define ULMBLAS_BLIS_SGEMM_MC_       168
#   define ULMBLAS_BLIS_SGEMM_KC_       256
#   define ULMBLAS_BLIS_SGEMM_NC_       4080
#   define ULMBLAS_BLIS_SGEMM_ALIGN     32
#   define ULMBLAS_BLIS_SGEMM_API       0
#   define ULMBLAS_BLIS_SGEMM_STORAGE   2
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   0
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_SGEMM_UKERNEL_blis_haswell_4x24)
//  BLIS 0.x kernel from kernels/haswell/3/old/: as haswell_24x4.
#   define ULMBLAS_BLIS_SGEMM_NAME      s_haswell_4x24
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  bli_sgemm_haswell_asm_4x24
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/s_haswell_4x24.h>
#   define ULMBLAS_BLIS_SGEMM_MR        4
#   define ULMBLAS_BLIS_SGEMM_NR        24
#   define ULMBLAS_BLIS_SGEMM_MC_       168
#   define ULMBLAS_BLIS_SGEMM_KC_       256
#   define ULMBLAS_BLIS_SGEMM_NC_       4080
#   define ULMBLAS_BLIS_SGEMM_ALIGN     32
#   define ULMBLAS_BLIS_SGEMM_API       0
#   define ULMBLAS_BLIS_SGEMM_STORAGE   2
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   1
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_SGEMM_UKERNEL_blis_sandybridge_8x8)
#   define ULMBLAS_BLIS_SGEMM_NAME      s_sandybridge_8x8
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  bli_sgemm_sandybridge_asm_8x8
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/s_sandybridge_8x8.h>
#   define ULMBLAS_BLIS_SGEMM_MR        8
#   define ULMBLAS_BLIS_SGEMM_NR        8
#   define ULMBLAS_BLIS_SGEMM_MC_       128     // BLIS sandybridge: 128, 384, 4096
#   define ULMBLAS_BLIS_SGEMM_KC_       384
#   define ULMBLAS_BLIS_SGEMM_NC_       4096
#   define ULMBLAS_BLIS_SGEMM_ALIGN     32
#   define ULMBLAS_BLIS_SGEMM_API       1
#   define ULMBLAS_BLIS_SGEMM_STORAGE   0       // GEMM_UKR_SETUP_CT, column major
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   0
#   if !defined(__AVX__)
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "AVX (e.g. -mavx)"
#elif defined(ULMBLAS_SGEMM_UKERNEL_blis_penryn_8x4)
#   define ULMBLAS_BLIS_SGEMM_NAME      s_penryn_8x4
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  bli_sgemm_penryn_asm_8x4
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/s_penryn_8x4.h>
#   define ULMBLAS_BLIS_SGEMM_MR        8
#   define ULMBLAS_BLIS_SGEMM_NR        4
#   define ULMBLAS_BLIS_SGEMM_MC_       768     // BLIS penryn: 768, 384, 4096
#   define ULMBLAS_BLIS_SGEMM_KC_       384
#   define ULMBLAS_BLIS_SGEMM_NC_       4096
#   define ULMBLAS_BLIS_SGEMM_ALIGN     16
#   define ULMBLAS_BLIS_SGEMM_API       1
#   define ULMBLAS_BLIS_SGEMM_STORAGE   0       // GEMM_UKR_SETUP_CT_ALIGNED (16 bytes)
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   0
#   if !defined(__SSE3__)
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "SSE3 (e.g. -msse3)"
#elif defined(ULMBLAS_SGEMM_UKERNEL_blis_skx_32x12_l2)
#   define ULMBLAS_BLIS_SGEMM_NAME      s_skx_32x12_l2
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  bli_sgemm_skx_asm_32x12_l2
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/s_skx_32x12_l2.h>
#   define ULMBLAS_BLIS_SGEMM_MR        32
#   define ULMBLAS_BLIS_SGEMM_NR        12
#   define ULMBLAS_BLIS_SGEMM_MC_       480     // BLIS skx: 480, 384, 3072
#   define ULMBLAS_BLIS_SGEMM_KC_       384
#   define ULMBLAS_BLIS_SGEMM_NC_       3072
#   define ULMBLAS_BLIS_SGEMM_ALIGN     64
#   define ULMBLAS_BLIS_SGEMM_API       1
#   define ULMBLAS_BLIS_SGEMM_STORAGE   0       // GEMM_UKR_SETUP_CT, column major
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   0
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#elif defined(ULMBLAS_SGEMM_UKERNEL_blis_aocl_haswell_6x16)
//  AOCL-BLAS's version of bli_sgemm_haswell_asm_6x16 (BLIS 0.x interface),
//  used by its zen, zen2, zen3 configurations; block sizes of zen3.
#   define ULMBLAS_BLIS_SGEMM_NAME      s_aocl_haswell_6x16
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  bli_sgemm_haswell_asm_6x16
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/s_aocl_haswell_6x16.h>
#   define ULMBLAS_BLIS_SGEMM_MR        6
#   define ULMBLAS_BLIS_SGEMM_NR        16
#   define ULMBLAS_BLIS_SGEMM_MC_       144     // AOCL 5.1 zen3: 144, 256, 4080
#   define ULMBLAS_BLIS_SGEMM_KC_       256
#   define ULMBLAS_BLIS_SGEMM_NC_       4080
#   define ULMBLAS_BLIS_SGEMM_ALIGN     32
#   define ULMBLAS_BLIS_SGEMM_API       0
#   define ULMBLAS_BLIS_SGEMM_STORAGE   2
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   1
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_SGEMM_UKERNEL_openblas_haswell_8x4)
//  OpenBLAS 0.3.34 HASWELL (sgemm_kernel_8x4_haswell_2.c): a kernel for a
//  whole packed block (16-row tiles from 2 panels of A, B in panels of 4
//  columns).
#   define ULMBLAS_BLIS_SGEMM_NAME      s_haswell_8x4
#   define ULMBLAS_BLIS_SGEMM_NAMESPACE openblas::kernel::s_haswell_8x4
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  sgemm_kernel_haswell
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/openblas/kernel/s_haswell_8x4.h>
#   define ULMBLAS_BLIS_SGEMM_MR        8
#   define ULMBLAS_BLIS_SGEMM_NR        4
//  OpenBLAS HASWELL: P 320, Q 320; R (sgemm_r) is computed at run time:
//  NC = 4096 blocks n <= 4096 as OpenBLAS does.
#   define ULMBLAS_BLIS_SGEMM_MC_       320
#   define ULMBLAS_BLIS_SGEMM_KC_       320
#   define ULMBLAS_BLIS_SGEMM_NC_       4096
#   define ULMBLAS_BLIS_SGEMM_ALIGN     32
#   define ULMBLAS_BLIS_SGEMM_API       2
#   define ULMBLAS_BLIS_SGEMM_STORAGE   0
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   0
#   define ULMBLAS_BLIS_SGEMM_BLOCKCALL 1
#   if !(defined(__AVX__) && defined(__AVX2__) && defined(__FMA__))
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "AVX2 and FMA (e.g. -mavx2 -mfma)"
#elif defined(ULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4)
//  OpenBLAS 0.3.34 SKYLAKEX (sgemm_kernel_16x4_skylakex_3.c): a kernel for
//  a whole packed block, B in panels of 4 columns.
#   define ULMBLAS_BLIS_SGEMM_NAME      s_skylakex_16x4
#   define ULMBLAS_BLIS_SGEMM_NAMESPACE openblas::kernel::s_skylakex_16x4
#   define ULMBLAS_BLIS_SGEMM_FUNCTION  sgemm_kernel_skylakex
#   define ULMBLAS_BLIS_SGEMM_HEADER    <ulmblas/level3/ukernel/openblas/kernel/s_skylakex_16x4.h>
#   define ULMBLAS_BLIS_SGEMM_MR        16
#   define ULMBLAS_BLIS_SGEMM_NR        4
//  OpenBLAS SKYLAKEX: P 448, Q 448; R as for HASWELL
#   define ULMBLAS_BLIS_SGEMM_MC_       448
#   define ULMBLAS_BLIS_SGEMM_KC_       448
#   define ULMBLAS_BLIS_SGEMM_NC_       4096
#   define ULMBLAS_BLIS_SGEMM_ALIGN     64
#   define ULMBLAS_BLIS_SGEMM_API       2
#   define ULMBLAS_BLIS_SGEMM_STORAGE   0
#   define ULMBLAS_BLIS_SGEMM_ROWPREF   0
#   define ULMBLAS_BLIS_SGEMM_BLOCKCALL 1
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_SGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_SGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#endif

#if defined(ULMBLAS_BLIS_SGEMM_NAME)

#   define USE_BLIS_SGEMM
#   define USE_BLIS_XGEMM

#   if (defined(ULMBLAS_SGEMM_UKERNEL_blis_haswell_6x16)                    \
      + defined(ULMBLAS_SGEMM_UKERNEL_blis_haswell_16x6)                    \
      + defined(ULMBLAS_SGEMM_UKERNEL_blis_haswell_24x4)                    \
      + defined(ULMBLAS_SGEMM_UKERNEL_blis_haswell_4x24)                    \
      + defined(ULMBLAS_SGEMM_UKERNEL_blis_sandybridge_8x8)                 \
      + defined(ULMBLAS_SGEMM_UKERNEL_blis_penryn_8x4)                      \
      + defined(ULMBLAS_SGEMM_UKERNEL_blis_skx_32x12_l2)                    \
      + defined(ULMBLAS_SGEMM_UKERNEL_blis_aocl_haswell_6x16)               \
      + defined(ULMBLAS_SGEMM_UKERNEL_openblas_haswell_8x4)                 \
      + defined(ULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4)) != 1
#       error "more than one ULMBLAS_SGEMM_UKERNEL_... defined"
#   endif

#   if !defined(__x86_64__) || defined(__ILP32__)
#       error "the vendored SGEMM micro kernels need x86-64 (LP64)"
#   endif

#   if defined(ULMBLAS_BLIS_SGEMM_ISA_MISSING)
#       error "the selected SGEMM micro kernel needs " ULMBLAS_BLIS_SGEMM_ISA
#   endif

#   if !defined(ULMBLAS_BLIS_SGEMM_NAMESPACE)
#       define ULMBLAS_BLIS_SGEMM_NAMESPACE blis::kernel::ULMBLAS_BLIS_SGEMM_NAME
#   endif
#   if !defined(ULMBLAS_BLIS_SGEMM_BLOCKCALL)
#       define ULMBLAS_BLIS_SGEMM_BLOCKCALL 0
#   endif

#   if defined(ULMBLAS_SGEMM_MC)
#       define ULMBLAS_BLIS_SGEMM_MC    ULMBLAS_SGEMM_MC
#   else
#       define ULMBLAS_BLIS_SGEMM_MC    ULMBLAS_BLIS_SGEMM_MC_
#   endif
#   if defined(ULMBLAS_SGEMM_KC)
#       define ULMBLAS_BLIS_SGEMM_KC    ULMBLAS_SGEMM_KC
#   else
#       define ULMBLAS_BLIS_SGEMM_KC    ULMBLAS_BLIS_SGEMM_KC_
#   endif
#   if defined(ULMBLAS_SGEMM_NC)
#       define ULMBLAS_BLIS_SGEMM_NC    ULMBLAS_SGEMM_NC
#   else
#       define ULMBLAS_BLIS_SGEMM_NC    ULMBLAS_BLIS_SGEMM_NC_
#   endif

#elif defined(ULMBLAS_SGEMM_MC) || defined(ULMBLAS_SGEMM_KC) \
   || defined(ULMBLAS_SGEMM_NC)
#   error "ULMBLAS_SGEMM_MC, KC, NC need a ULMBLAS_SGEMM_UKERNEL_... kernel"
#endif

#endif // ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_S_H
