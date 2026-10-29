#ifndef ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_H
#define ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_H 1

//
//  Selection of a BLIS-derived DGEMM micro kernel at build time.
//
//  By default ulmBLAS uses its own kernel selection (ulmblas/config/simd.h:
//  reference, SSE, AVX/FMA, NEON, SME2).  Defining exactly one macro
//
//      ULMBLAS_DGEMM_UKERNEL_blis_NAME     (e.g. -DULMBLAS_DGEMM_UKERNEL_blis_skx_16x14)
//
//  replaces the micro kernel for double by the vendored BLIS kernel NAME
//  (kernel/NAME.h, see README.md).  All other data types keep the kernel
//  that would be selected without it.  The compiler flags must enable the
//  instruction set of the kernel (e.g. ARCHFLAGS="-mavx -mfma -mavx512f"
//  for the AVX-512 kernels); the CPU must support it at run time.
//
//  For each kernel:
//
//      ..._MR, ..._NR   register block size (fixed by the kernel)
//      ..._MC, KC, NC   default cache block sizes: the values of the BLIS
//                       configuration that uses the kernel, adjusted to the
//                       conditions of ulmBLAS (MC a multiple of MR and NR,
//                       NC a multiple of NR); -DULMBLAS_DGEMM_MC=... etc.
//                       override them
//      ..._ALIGN        alignment (bytes) the kernel needs for the packed
//                       panels (aligned loads of A or B)
//      ..._API          1: BLIS 1.0 interface (m, n, k, ..., void pointers,
//                       edge cases handled by the kernel), 0: interface of
//                       BLIS 0.x (k, ..., double pointers, full tiles only),
//                       2: OpenBLAS kernel (m, n, k, alpha, A, B, C, ldc;
//                       C <- C + alpha*A*B, the adapter applies beta)
//      ..._NAMESPACE    namespace of the kernel function
//      ..._STORAGE      storage of C the assembly code supports (others go
//                       through a local tile in the adapter, see ugemm.h):
//                       0 column major (rs_c = 1), 1 column or row major,
//                       2 any strides, 3 row major (cs_c = 1) only,
//                       4 column major with 32-byte aligned columns
//      ..._KMAX         OpenBLAS assembly kernels: largest k per call (the
//                       adapter splits longer k loops, see ugemm.h)
//      ..._BLOCKCALL    OpenBLAS kernels whose code paths address B for any
//                       number of full panels: the macro kernel calls them
//                       once for all full tiles of a block (ugemm_block in
//                       ugemm.h), otherwise once per panel of B
//      ..._INTERLEAVE_PANELS
//                       number of panels of B that DGEMM packs at once when
//                       it packs B interleaved with the first block of A
//                       (gemm.tcc; default 1)
//
#if defined(ULMBLAS_DGEMM_UKERNEL_blis_haswell_6x8)
#   define ULMBLAS_BLIS_DGEMM_NAME      haswell_6x8
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_haswell_asm_6x8
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/haswell_6x8.h>
#   define ULMBLAS_BLIS_DGEMM_MR        6
#   define ULMBLAS_BLIS_DGEMM_NR        8
#   define ULMBLAS_BLIS_DGEMM_MC_       72      // BLIS haswell: 72, 256, 4080
#   define ULMBLAS_BLIS_DGEMM_KC_       256
#   define ULMBLAS_BLIS_DGEMM_NC_       4080
#   define ULMBLAS_BLIS_DGEMM_ALIGN     32
#   define ULMBLAS_BLIS_DGEMM_API       1
#   define ULMBLAS_BLIS_DGEMM_STORAGE   1
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_haswell_8x6)
#   define ULMBLAS_BLIS_DGEMM_NAME      haswell_8x6
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_haswell_asm_8x6
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/haswell_8x6.h>
#   define ULMBLAS_BLIS_DGEMM_MR        8
#   define ULMBLAS_BLIS_DGEMM_NR        6
#   define ULMBLAS_BLIS_DGEMM_MC_       72      // BLIS haswell (8 x 6): 72, 256, 4080
#   define ULMBLAS_BLIS_DGEMM_KC_       256
#   define ULMBLAS_BLIS_DGEMM_NC_       4080
#   define ULMBLAS_BLIS_DGEMM_ALIGN     32
#   define ULMBLAS_BLIS_DGEMM_API       1
#   define ULMBLAS_BLIS_DGEMM_STORAGE   1
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_haswell_12x4)
//  BLIS 0.x kernel from kernels/haswell/3/old/ (not used by any BLIS
//  configuration): block sizes of the haswell configuration.
#   define ULMBLAS_BLIS_DGEMM_NAME      haswell_12x4
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_haswell_asm_12x4
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/haswell_12x4.h>
#   define ULMBLAS_BLIS_DGEMM_MR        12
#   define ULMBLAS_BLIS_DGEMM_NR        4
#   define ULMBLAS_BLIS_DGEMM_MC_       72
#   define ULMBLAS_BLIS_DGEMM_KC_       256
#   define ULMBLAS_BLIS_DGEMM_NC_       4080
#   define ULMBLAS_BLIS_DGEMM_ALIGN     32
#   define ULMBLAS_BLIS_DGEMM_API       0
#   define ULMBLAS_BLIS_DGEMM_STORAGE   2
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_haswell_4x12)
//  BLIS 0.x kernel from kernels/haswell/3/old/ (not used by any BLIS
//  configuration): block sizes of the haswell configuration.
#   define ULMBLAS_BLIS_DGEMM_NAME      haswell_4x12
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_haswell_asm_4x12
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/haswell_4x12.h>
#   define ULMBLAS_BLIS_DGEMM_MR        4
#   define ULMBLAS_BLIS_DGEMM_NR        12
#   define ULMBLAS_BLIS_DGEMM_MC_       72
#   define ULMBLAS_BLIS_DGEMM_KC_       256
#   define ULMBLAS_BLIS_DGEMM_NC_       4080
#   define ULMBLAS_BLIS_DGEMM_ALIGN     32
#   define ULMBLAS_BLIS_DGEMM_API       0
#   define ULMBLAS_BLIS_DGEMM_STORAGE   2
#   if !(defined(__AVX__) && defined(__FMA__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX and FMA (e.g. -mavx -mfma)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_sandybridge_8x4)
#   define ULMBLAS_BLIS_DGEMM_NAME      sandybridge_8x4
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_sandybridge_asm_8x4
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/sandybridge_8x4.h>
#   define ULMBLAS_BLIS_DGEMM_MR        8
#   define ULMBLAS_BLIS_DGEMM_NR        4
#   define ULMBLAS_BLIS_DGEMM_MC_       96      // BLIS sandybridge: 96, 256, 4096
#   define ULMBLAS_BLIS_DGEMM_KC_       256
#   define ULMBLAS_BLIS_DGEMM_NC_       4096
#   define ULMBLAS_BLIS_DGEMM_ALIGN     32
#   define ULMBLAS_BLIS_DGEMM_API       1
#   define ULMBLAS_BLIS_DGEMM_STORAGE   0
#   if !defined(__AVX__)
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX (e.g. -mavx)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_sandybridge_int_8x4)
//  The intrinsics version of sandybridge_8x4 (not used by the sandybridge
//  configuration, which uses the assembly kernel): same block sizes.
#   define ULMBLAS_BLIS_DGEMM_NAME      sandybridge_int_8x4
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_sandybridge_int_8x4
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/sandybridge_int_8x4.h>
#   define ULMBLAS_BLIS_DGEMM_MR        8
#   define ULMBLAS_BLIS_DGEMM_NR        4
#   define ULMBLAS_BLIS_DGEMM_MC_       96
#   define ULMBLAS_BLIS_DGEMM_KC_       256
#   define ULMBLAS_BLIS_DGEMM_NC_       4096
#   define ULMBLAS_BLIS_DGEMM_ALIGN     32
#   define ULMBLAS_BLIS_DGEMM_API       1
//  It stores C with _mm256_store_pd (32-byte alignment), although it uses
//  GEMM_UKR_SETUP_CT (not ..._ALIGNED): other C goes through a local tile.
#   define ULMBLAS_BLIS_DGEMM_STORAGE   4
#   if !defined(__AVX__)
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX (e.g. -mavx)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_penryn_4x4)
#   define ULMBLAS_BLIS_DGEMM_NAME      penryn_4x4
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_penryn_asm_4x4
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/penryn_4x4.h>
#   define ULMBLAS_BLIS_DGEMM_MR        4
#   define ULMBLAS_BLIS_DGEMM_NR        4
#   define ULMBLAS_BLIS_DGEMM_MC_       384     // BLIS penryn: 384, 384, 4096
#   define ULMBLAS_BLIS_DGEMM_KC_       384
#   define ULMBLAS_BLIS_DGEMM_NC_       4096
#   define ULMBLAS_BLIS_DGEMM_ALIGN     16
#   define ULMBLAS_BLIS_DGEMM_API       1
#   define ULMBLAS_BLIS_DGEMM_STORAGE   0
#   if !defined(__SSE3__)
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "SSE3 (e.g. -msse3)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_skx_16x14)
#   define ULMBLAS_BLIS_DGEMM_NAME      skx_16x14
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_skx_asm_16x14
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/skx_16x14.h>
#   define ULMBLAS_BLIS_DGEMM_MR        16
#   define ULMBLAS_BLIS_DGEMM_NR        14
#   define ULMBLAS_BLIS_DGEMM_MC_       224     // BLIS skx: 240 (not a multiple of 14), 256, 3752
#   define ULMBLAS_BLIS_DGEMM_KC_       256
#   define ULMBLAS_BLIS_DGEMM_NC_       3752
#   define ULMBLAS_BLIS_DGEMM_ALIGN     64
#   define ULMBLAS_BLIS_DGEMM_API       1
#   define ULMBLAS_BLIS_DGEMM_STORAGE   0
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_skx_16x12_l2)
#   define ULMBLAS_BLIS_DGEMM_NAME      skx_16x12_l2
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_skx_asm_16x12_l2
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/skx_16x12_l2.h>
#   define ULMBLAS_BLIS_DGEMM_MR        16
#   define ULMBLAS_BLIS_DGEMM_NR        12
#   define ULMBLAS_BLIS_DGEMM_MC_       240     // not used by a BLIS configuration: skx values,
#   define ULMBLAS_BLIS_DGEMM_KC_       256     // NC 3752 rounded to a multiple of 12
#   define ULMBLAS_BLIS_DGEMM_NC_       3744
#   define ULMBLAS_BLIS_DGEMM_ALIGN     64
#   define ULMBLAS_BLIS_DGEMM_API       1
#   define ULMBLAS_BLIS_DGEMM_STORAGE   0
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_zen4_32x6)
#   define ULMBLAS_BLIS_DGEMM_NAME      zen4_32x6
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_zen4_asm_32x6
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/zen4_32x6.h>
#   define ULMBLAS_BLIS_DGEMM_MR        32
#   define ULMBLAS_BLIS_DGEMM_NR        6
#   define ULMBLAS_BLIS_DGEMM_MC_       96      // AOCL zen4 (Genoa): 128 (not a multiple of 6), 512, 4002
#   define ULMBLAS_BLIS_DGEMM_KC_       512
#   define ULMBLAS_BLIS_DGEMM_NC_       4002
#   define ULMBLAS_BLIS_DGEMM_ALIGN     64
#   define ULMBLAS_BLIS_DGEMM_API       0
#   define ULMBLAS_BLIS_DGEMM_STORAGE   2
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_zen4_8x24)
#   define ULMBLAS_BLIS_DGEMM_NAME      zen4_8x24
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_zen4_asm_8x24
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/zen4_8x24.h>
#   define ULMBLAS_BLIS_DGEMM_MR        8
#   define ULMBLAS_BLIS_DGEMM_NR        24
#   define ULMBLAS_BLIS_DGEMM_MC_       120     // AOCL zen4, TRSM block sizes: 120, 512, 4008
#   define ULMBLAS_BLIS_DGEMM_KC_       512
#   define ULMBLAS_BLIS_DGEMM_NC_       4008
#   define ULMBLAS_BLIS_DGEMM_ALIGN     64
#   define ULMBLAS_BLIS_DGEMM_API       0
#   define ULMBLAS_BLIS_DGEMM_STORAGE   2
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12)
//  Not a BLIS kernel: the DGEMM kernel of OpenBLAS 0.3.34 for SKYLAKEX, see
//  ulmblas/level3/ukernel/openblas/.
#   define ULMBLAS_BLIS_DGEMM_NAME      skylakex_16x12
#   define ULMBLAS_BLIS_DGEMM_NAMESPACE openblas::kernel::skylakex_16x12
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  dgemm_kernel_skylakex
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/openblas/kernel/skylakex_16x12.h>
#   define ULMBLAS_BLIS_DGEMM_MR        16
#   define ULMBLAS_BLIS_DGEMM_NR        12
#   define ULMBLAS_BLIS_DGEMM_MC_       192     // OpenBLAS SKYLAKEX: P 192, Q 384, R 8640
#   define ULMBLAS_BLIS_DGEMM_KC_       384
#   define ULMBLAS_BLIS_DGEMM_NC_       8640
#   define ULMBLAS_BLIS_DGEMM_ALIGN     64
#   define ULMBLAS_BLIS_DGEMM_API       2
#   define ULMBLAS_BLIS_DGEMM_STORAGE   0
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8)
//  Not a BLIS kernel: the DGEMM kernel of OpenBLAS 0.3.34 for HASWELL (an
//  assembly file, ulmblas/level3/ukernel/openblas/asm/haswell_4x8.S, built
//  into the libraries by make.inc), see ulmblas/level3/ukernel/openblas/.
#   define ULMBLAS_BLIS_DGEMM_NAME      haswell_4x8
#   define ULMBLAS_BLIS_DGEMM_NAMESPACE openblas::kernel::haswell_4x8
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  dgemm_kernel
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/openblas/kernel/haswell_4x8.h>
#   define ULMBLAS_BLIS_DGEMM_MR        4
#   define ULMBLAS_BLIS_DGEMM_NR        8
#   define ULMBLAS_BLIS_DGEMM_MC_       512     // OpenBLAS HASWELL: P 512, Q 256, R 13824
#   define ULMBLAS_BLIS_DGEMM_KC_       256
#   define ULMBLAS_BLIS_DGEMM_NC_       13824
#   define ULMBLAS_BLIS_DGEMM_ALIGN     32
#   define ULMBLAS_BLIS_DGEMM_API       2
#   define ULMBLAS_BLIS_DGEMM_STORAGE   0
//  The kernel copies 12 columns of B into a stack buffer of 256*8*12 + 4096
//  bytes (k <= 298); OpenBLAS calls it with k <= Q = 256.
#   define ULMBLAS_BLIS_DGEMM_KMAX      256
//  OpenBLAS's driver calls the kernel for whole blocks and, for the first
//  block of A, for 3*DGEMM_UNROLL_N = 24 columns of B at a time: the
//  kernel computes 4 x 12 tiles only for groups of 24 columns.
#   define ULMBLAS_BLIS_DGEMM_BLOCKCALL 1
#   define ULMBLAS_BLIS_DGEMM_INTERLEAVE_PANELS 3
#   if !(defined(__AVX__) && defined(__AVX2__) && defined(__FMA__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX2 and FMA (e.g. -mavx2 -mfma)"
#elif defined(ULMBLAS_DGEMM_UKERNEL_blis_zen5_8x24)
#   define ULMBLAS_BLIS_DGEMM_NAME      zen5_8x24
#   define ULMBLAS_BLIS_DGEMM_FUNCTION  bli_dgemm_avx512_asm_8x24
#   define ULMBLAS_BLIS_DGEMM_HEADER    <ulmblas/level3/ukernel/blis/kernel/zen5_8x24.h>
#   define ULMBLAS_BLIS_DGEMM_MR        8
#   define ULMBLAS_BLIS_DGEMM_NR        24
#   define ULMBLAS_BLIS_DGEMM_MC_       120     // AOCL zen5 (5.0, 5.1): 120, 512, 2016
#   define ULMBLAS_BLIS_DGEMM_KC_       512
#   define ULMBLAS_BLIS_DGEMM_NC_       2016
#   define ULMBLAS_BLIS_DGEMM_ALIGN     64
#   define ULMBLAS_BLIS_DGEMM_API       0
//  From AOCL-BLAS 5.1: in 5.0 the column-major and general-stride paths of
//  this kernel compared beta with a register holding elements of B.
#   define ULMBLAS_BLIS_DGEMM_STORAGE   2
#   if !(defined(__AVX__) && defined(__FMA__) && defined(__AVX512F__))
#       define ULMBLAS_BLIS_DGEMM_ISA_MISSING
#   endif
#   define ULMBLAS_BLIS_DGEMM_ISA       "AVX, FMA and AVX-512F (e.g. -mavx -mfma -mavx512f)"
#endif

#if defined(ULMBLAS_BLIS_DGEMM_NAME)

#   define USE_BLIS_DGEMM

#   if (defined(ULMBLAS_DGEMM_UKERNEL_blis_haswell_6x8)                     \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_haswell_8x6)                     \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_haswell_12x4)                    \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_haswell_4x12)                    \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_sandybridge_8x4)                 \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_sandybridge_int_8x4)             \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_penryn_4x4)                      \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_skx_16x14)                       \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_skx_16x12_l2)                    \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_zen4_32x6)                       \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_zen4_8x24)                       \
      + defined(ULMBLAS_DGEMM_UKERNEL_blis_zen5_8x24)                       \
      + defined(ULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12)               \
      + defined(ULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8)) != 1
#       error "more than one ULMBLAS_DGEMM_UKERNEL_... defined"
#   endif

#   if !defined(__x86_64__) || defined(__ILP32__)
#       error "the BLIS DGEMM micro kernels need x86-64 (LP64)"
#   endif

#   if defined(ULMBLAS_BLIS_DGEMM_ISA_MISSING)
#       error "the selected BLIS DGEMM micro kernel needs " ULMBLAS_BLIS_DGEMM_ISA
#   endif

#   if !defined(ULMBLAS_BLIS_DGEMM_NAMESPACE)
#       define ULMBLAS_BLIS_DGEMM_NAMESPACE blis::kernel::ULMBLAS_BLIS_DGEMM_NAME
#   endif

#   if defined(ULMBLAS_DGEMM_MC)
#       define ULMBLAS_BLIS_DGEMM_MC    ULMBLAS_DGEMM_MC
#   else
#       define ULMBLAS_BLIS_DGEMM_MC    ULMBLAS_BLIS_DGEMM_MC_
#   endif
#   if defined(ULMBLAS_DGEMM_KC)
#       define ULMBLAS_BLIS_DGEMM_KC    ULMBLAS_DGEMM_KC
#   else
#       define ULMBLAS_BLIS_DGEMM_KC    ULMBLAS_BLIS_DGEMM_KC_
#   endif
#   if defined(ULMBLAS_DGEMM_NC)
#       define ULMBLAS_BLIS_DGEMM_NC    ULMBLAS_DGEMM_NC
#   else
#       define ULMBLAS_BLIS_DGEMM_NC    ULMBLAS_BLIS_DGEMM_NC_
#   endif

#elif defined(ULMBLAS_DGEMM_MC) || defined(ULMBLAS_DGEMM_KC) \
   || defined(ULMBLAS_DGEMM_NC)
#   error "ULMBLAS_DGEMM_MC, KC, NC need a ULMBLAS_DGEMM_UKERNEL_blis_... kernel"
#endif

#endif // ULMBLAS_LEVEL3_UKERNEL_BLIS_SELECT_H
