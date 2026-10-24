/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


*/

//
//  ulmBLAS: AArch64 NEON micro kernel for double (MR = 6, NR = 8).
//
//  Provenance: the inline assembly is the micro kernel
//  bli_dgemm_armv8a_asm_8x6r of BLIS (https://github.com/flame/blis, file
//  kernels/armv8a/3/bli_gemm_armv8a_asm_d8x6r.c with the macros of
//  armv8a_asm_utils.h and armv8a_asm_d2x2.h, tag 2.0, commit
//  e8566eb3e773fb54d11b33e371d13f22d2941e50), the double precision kernel of
//  the BLIS configurations firestorm (Apple M1) and armv8a.  The macros are
//  expanded.
//
//  The BLIS kernel computes an 8 x 6 tile with unit column stride (rows of
//  the tile in the NEON registers v0 to v23).  ulmBLAS calls it for the
//  transposed product: for C with unit row stride, C^T = B^T A^T is an 8 x 6
//  tile with unit column stride, so the packed panel of B (NR = 8) is the
//  operand a of the BLIS kernel and the packed panel of A (MR = 6) the
//  operand b (in the comments of the assembly code a and b are the operands
//  of the BLIS kernel).
//
//  Modifications for ulmBLAS (the computational instructions are those of
//  BLIS):
//
//  - Interface of the ulmBLAS micro kernels (alpha and beta by reference,
//    strides of C as IndexType); operands swapped as described above.
//  - Numeric local labels (1: ... with 1b, 1f) instead of the named labels
//    (BLIS: "L<name>%=" for clang, ".<name>" for GCC), the same for all
//    compilers; see issue #0007 in test/known_issues.md for named labels on
//    Mach-O.
//  - The test of the flag of BLIS's temporary tile (GEMM_UKR_SETUP_CT) is
//    removed: the assembly code is only called for C with unit row stride,
//    for other strides (BLIS: the temporary tile) alpha*A*B is computed in a
//    local column major tile and C is updated in C++.  The prefetches of C
//    are unconditional.
//  - "cc" and "memory" are in the clobber list (the assembly code writes C
//    and the condition flags).
//
//  nextA and nextB are prefetched as in BLIS (operands swapped).  alpha ==
//  1 skips the scaling, beta == 0 skips reading C; NaN is neither 1 nor 0
//  (fcmp sets the flags as unordered), so alpha = NaN and beta = NaN are
//  computed as for any other value.
//
//  Requirements: AArch64 (selected with USE_NEON, see
//  ulmblas/config/simd.h).  No alignment of the panels is required.  A and
//  B are read exactly within the panels (MR*kc and NR*kc elements).
//

#ifndef ULMBLAS_LEVEL3_UKERNEL_NEON_UGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_NEON_UGEMM_TCC 1

#include <cstdint>
#include <ulmblas/level3/ukernel/neon/ugemm.h>

namespace ulmBLAS { namespace neon {

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
void>::type
ugemm(IndexType      kc,
      const double   &alpha,
      const double   *A,
      const double   *B,
      const double   &beta,
      double         *C,
      IndexType      incRowC,
      IndexType      incColC,
      const double   *nextA,
      const double   *nextB)
{
    const IndexType MR = BlockSizeUGemm<double>::MR;
    const IndexType NR = BlockSizeUGemm<double>::NR;

    if (incRowC!=IndexType(1)) {
        double C_[MR*NR];

        ugemm(kc, alpha, A, B, 0.0, C_, IndexType(1), MR, nextA, nextB);
        if (beta==0.0) {
            for (IndexType j=0; j<NR; ++j) {
                for (IndexType i=0; i<MR; ++i) {
                    C[i*incRowC+j*incColC] = C_[i+j*MR];
                }
            }
        } else {
            for (IndexType j=0; j<NR; ++j) {
                for (IndexType i=0; i<MR; ++i) {
                    C[i*incRowC+j*incColC] = beta*C[i*incRowC+j*incColC]
                                           + C_[i+j*MR];
                }
            }
        }
        return;
    }

    //  Operands of the BLIS kernel: a = B, b = A, tile = C^T with row stride
    //  incColC.
    const double  *a      = B;
    const double  *b      = A;
    const double  *a_next = nextB;
    const double  *b_next = nextA;
    uint64_t      k_mker  = kc / 4;
    uint64_t      k_left  = kc % 4;
    uint64_t      rs_c    = incColC;
    const double  *alpha_ = &alpha;
    const double  *beta_  = &beta;

    __asm__ volatile (
    "ldr     x0, %[a]\n\t"
    "ldr     x1, %[b]\n\t"
    "mov     x2, #8\n\t"
    "mov     x3, #6\n\t"
    "ldr     x5, %[c]\n\t"
    "ldr     x6, %[rs_c]\n\t"
    "lsl     x2, x2, #3\n\t"
    "lsl     x3, x3, #3\n\t"
    "lsl     x6, x6, #3\n\t"
    "mov     x9, x5\n\t"
    "prfm    PLDL1KEEP, [x9]\n\t"
    "prfm    PLDL1KEEP, [x9, 40]\n\t"
    "add     x9, x9, x6\n\t"
    "prfm    PLDL1KEEP, [x9]\n\t"
    "prfm    PLDL1KEEP, [x9, 40]\n\t"
    "add     x9, x9, x6\n\t"
    "prfm    PLDL1KEEP, [x9]\n\t"
    "prfm    PLDL1KEEP, [x9, 40]\n\t"
    "add     x9, x9, x6\n\t"
    "prfm    PLDL1KEEP, [x9]\n\t"
    "prfm    PLDL1KEEP, [x9, 40]\n\t"
    "add     x9, x9, x6\n\t"
    "prfm    PLDL1KEEP, [x9]\n\t"
    "prfm    PLDL1KEEP, [x9, 40]\n\t"
    "add     x9, x9, x6\n\t"
    "prfm    PLDL1KEEP, [x9]\n\t"
    "prfm    PLDL1KEEP, [x9, 40]\n\t"
    "add     x9, x9, x6\n\t"
    "prfm    PLDL1KEEP, [x9]\n\t"
    "prfm    PLDL1KEEP, [x9, 40]\n\t"
    "add     x9, x9, x6\n\t"
    "prfm    PLDL1KEEP, [x9]\n\t"
    "prfm    PLDL1KEEP, [x9, 40]\n\t"
    "add     x9, x9, x6\n\t"
    "ldr     x4, %[k_mker]\n\t"
    "ldr     x8, %[k_left]\n\t"

    //  Load the first column of a and row of b (if k >= 4), clear the tile
    "1:\n\t"
    "cmp     x4, #0\n\t"
    "b.eq    2f\n\t"
    "ldr     q24, [x0, #16*0]\n\t"
    "ldr     q25, [x0, #16*1]\n\t"
    "ldr     q26, [x0, #16*2]\n\t"
    "ldr     q27, [x0, #16*3]\n\t"
    "add     x0, x0, x2\n\t"
    "ldr     q28, [x1, #16*0]\n\t"
    "ldr     q29, [x1, #16*1]\n\t"
    "ldr     q30, [x1, #16*2]\n\t"
    "add     x1, x1, x3\n\t"
    "ldr     q31, [x1, #16*0]\n\t"
    "2:\n\t"
    "dup     v0.2d, xzr\n\t"
    "dup     v1.2d, xzr\n\t"
    "dup     v2.2d, xzr\n\t"
    "dup     v3.2d, xzr\n\t"
    "dup     v4.2d, xzr\n\t"
    "dup     v5.2d, xzr\n\t"
    "dup     v6.2d, xzr\n\t"
    "dup     v7.2d, xzr\n\t"
    "dup     v8.2d, xzr\n\t"
    "dup     v9.2d, xzr\n\t"
    "dup     v10.2d, xzr\n\t"
    "dup     v11.2d, xzr\n\t"
    "dup     v12.2d, xzr\n\t"
    "dup     v13.2d, xzr\n\t"
    "dup     v14.2d, xzr\n\t"
    "dup     v15.2d, xzr\n\t"
    "dup     v16.2d, xzr\n\t"
    "dup     v17.2d, xzr\n\t"
    "dup     v18.2d, xzr\n\t"
    "dup     v19.2d, xzr\n\t"
    "dup     v20.2d, xzr\n\t"
    "dup     v21.2d, xzr\n\t"
    "dup     v22.2d, xzr\n\t"
    "dup     v23.2d, xzr\n\t"
    "b.eq    5f\n\t"

    //  Main loop: 4 rank-1 updates per iteration, the loads of the next
    //  column of a and row of b interleaved with the FMAs
    "3:\n\t"
    "fmla    v0.2d, v28.2d, v24.d[0]\n\t"
    "fmla    v3.2d, v28.2d, v24.d[1]\n\t"
    "fmla    v6.2d, v28.2d, v25.d[0]\n\t"
    "fmla    v9.2d, v28.2d, v25.d[1]\n\t"
    "fmla    v1.2d, v29.2d, v24.d[0]\n\t"
    "fmla    v4.2d, v29.2d, v24.d[1]\n\t"
    "fmla    v7.2d, v29.2d, v25.d[0]\n\t"
    "fmla    v10.2d, v29.2d, v25.d[1]\n\t"
    "fmla    v2.2d, v30.2d, v24.d[0]\n\t"
    "fmla    v5.2d, v30.2d, v24.d[1]\n\t"
    "fmla    v8.2d, v30.2d, v25.d[0]\n\t"
    "fmla    v11.2d, v30.2d, v25.d[1]\n\t"
    "ldr     q24, [x0, #0]\n\t"
    "ldr     q25, [x0, #0+16]\n\t"
    "fmla    v12.2d, v28.2d, v26.d[0]\n\t"
    "fmla    v15.2d, v28.2d, v26.d[1]\n\t"
    "fmla    v18.2d, v28.2d, v27.d[0]\n\t"
    "fmla    v21.2d, v28.2d, v27.d[1]\n\t"
    "ldr     q28, [x1, #16]\n\t"
    "fmla    v13.2d, v29.2d, v26.d[0]\n\t"
    "fmla    v16.2d, v29.2d, v26.d[1]\n\t"
    "fmla    v19.2d, v29.2d, v27.d[0]\n\t"
    "fmla    v22.2d, v29.2d, v27.d[1]\n\t"
    "ldr     q29, [x1, #16+16]\n\t"
    "fmla    v14.2d, v30.2d, v26.d[0]\n\t"
    "fmla    v17.2d, v30.2d, v26.d[1]\n\t"
    "fmla    v20.2d, v30.2d, v27.d[0]\n\t"
    "fmla    v23.2d, v30.2d, v27.d[1]\n\t"
    "add     x1, x1, x3\n\t"
    "ldr     q30, [x1, #16*0]\n\t"
    "ldr     q26, [x0, #16*2]\n\t"
    "ldr     q27, [x0, #16*3]\n\t"
    "add     x0, x0, x2\n\t"
    "fmla    v0.2d, v31.2d, v24.d[0]\n\t"
    "fmla    v3.2d, v31.2d, v24.d[1]\n\t"
    "fmla    v6.2d, v31.2d, v25.d[0]\n\t"
    "fmla    v9.2d, v31.2d, v25.d[1]\n\t"
    "fmla    v1.2d, v28.2d, v24.d[0]\n\t"
    "fmla    v4.2d, v28.2d, v24.d[1]\n\t"
    "fmla    v7.2d, v28.2d, v25.d[0]\n\t"
    "fmla    v10.2d, v28.2d, v25.d[1]\n\t"
    "fmla    v2.2d, v29.2d, v24.d[0]\n\t"
    "fmla    v5.2d, v29.2d, v24.d[1]\n\t"
    "fmla    v8.2d, v29.2d, v25.d[0]\n\t"
    "fmla    v11.2d, v29.2d, v25.d[1]\n\t"
    "ldr     q24, [x0, #0]\n\t"
    "ldr     q25, [x0, #0+16]\n\t"
    "fmla    v12.2d, v31.2d, v26.d[0]\n\t"
    "fmla    v15.2d, v31.2d, v26.d[1]\n\t"
    "fmla    v18.2d, v31.2d, v27.d[0]\n\t"
    "fmla    v21.2d, v31.2d, v27.d[1]\n\t"
    "ldr     q31, [x1, #16]\n\t"
    "fmla    v13.2d, v28.2d, v26.d[0]\n\t"
    "fmla    v16.2d, v28.2d, v26.d[1]\n\t"
    "fmla    v19.2d, v28.2d, v27.d[0]\n\t"
    "fmla    v22.2d, v28.2d, v27.d[1]\n\t"
    "ldr     q28, [x1, #16+16]\n\t"
    "fmla    v14.2d, v29.2d, v26.d[0]\n\t"
    "fmla    v17.2d, v29.2d, v26.d[1]\n\t"
    "fmla    v20.2d, v29.2d, v27.d[0]\n\t"
    "fmla    v23.2d, v29.2d, v27.d[1]\n\t"
    "add     x1, x1, x3\n\t"
    "ldr     q29, [x1, #16*0]\n\t"
    "ldr     q26, [x0, #16*2]\n\t"
    "ldr     q27, [x0, #16*3]\n\t"
    "add     x0, x0, x2\n\t"
    "subs    x4, x4, #1\n\t"
    "b.eq    4f\n\t"
    "fmla    v0.2d, v30.2d, v24.d[0]\n\t"
    "fmla    v3.2d, v30.2d, v24.d[1]\n\t"
    "fmla    v6.2d, v30.2d, v25.d[0]\n\t"
    "fmla    v9.2d, v30.2d, v25.d[1]\n\t"
    "fmla    v1.2d, v31.2d, v24.d[0]\n\t"
    "fmla    v4.2d, v31.2d, v24.d[1]\n\t"
    "fmla    v7.2d, v31.2d, v25.d[0]\n\t"
    "fmla    v10.2d, v31.2d, v25.d[1]\n\t"
    "fmla    v2.2d, v28.2d, v24.d[0]\n\t"
    "fmla    v5.2d, v28.2d, v24.d[1]\n\t"
    "fmla    v8.2d, v28.2d, v25.d[0]\n\t"
    "fmla    v11.2d, v28.2d, v25.d[1]\n\t"
    "ldr     q24, [x0, #0]\n\t"
    "ldr     q25, [x0, #0+16]\n\t"
    "fmla    v12.2d, v30.2d, v26.d[0]\n\t"
    "fmla    v15.2d, v30.2d, v26.d[1]\n\t"
    "fmla    v18.2d, v30.2d, v27.d[0]\n\t"
    "fmla    v21.2d, v30.2d, v27.d[1]\n\t"
    "ldr     q30, [x1, #16]\n\t"
    "fmla    v13.2d, v31.2d, v26.d[0]\n\t"
    "fmla    v16.2d, v31.2d, v26.d[1]\n\t"
    "fmla    v19.2d, v31.2d, v27.d[0]\n\t"
    "fmla    v22.2d, v31.2d, v27.d[1]\n\t"
    "ldr     q31, [x1, #16+16]\n\t"
    "fmla    v14.2d, v28.2d, v26.d[0]\n\t"
    "fmla    v17.2d, v28.2d, v26.d[1]\n\t"
    "fmla    v20.2d, v28.2d, v27.d[0]\n\t"
    "fmla    v23.2d, v28.2d, v27.d[1]\n\t"
    "add     x1, x1, x3\n\t"
    "ldr     q28, [x1, #16*0]\n\t"
    "ldr     q26, [x0, #16*2]\n\t"
    "ldr     q27, [x0, #16*3]\n\t"
    "add     x0, x0, x2\n\t"
    "fmla    v0.2d, v29.2d, v24.d[0]\n\t"
    "fmla    v3.2d, v29.2d, v24.d[1]\n\t"
    "fmla    v6.2d, v29.2d, v25.d[0]\n\t"
    "fmla    v9.2d, v29.2d, v25.d[1]\n\t"
    "fmla    v1.2d, v30.2d, v24.d[0]\n\t"
    "fmla    v4.2d, v30.2d, v24.d[1]\n\t"
    "fmla    v7.2d, v30.2d, v25.d[0]\n\t"
    "fmla    v10.2d, v30.2d, v25.d[1]\n\t"
    "fmla    v2.2d, v31.2d, v24.d[0]\n\t"
    "fmla    v5.2d, v31.2d, v24.d[1]\n\t"
    "fmla    v8.2d, v31.2d, v25.d[0]\n\t"
    "fmla    v11.2d, v31.2d, v25.d[1]\n\t"
    "ldr     q24, [x0, #0]\n\t"
    "ldr     q25, [x0, #0+16]\n\t"
    "fmla    v12.2d, v29.2d, v26.d[0]\n\t"
    "fmla    v15.2d, v29.2d, v26.d[1]\n\t"
    "fmla    v18.2d, v29.2d, v27.d[0]\n\t"
    "fmla    v21.2d, v29.2d, v27.d[1]\n\t"
    "ldr     q29, [x1, #16]\n\t"
    "fmla    v13.2d, v30.2d, v26.d[0]\n\t"
    "fmla    v16.2d, v30.2d, v26.d[1]\n\t"
    "fmla    v19.2d, v30.2d, v27.d[0]\n\t"
    "fmla    v22.2d, v30.2d, v27.d[1]\n\t"
    "ldr     q30, [x1, #16+16]\n\t"
    "fmla    v14.2d, v31.2d, v26.d[0]\n\t"
    "fmla    v17.2d, v31.2d, v26.d[1]\n\t"
    "fmla    v20.2d, v31.2d, v27.d[0]\n\t"
    "fmla    v23.2d, v31.2d, v27.d[1]\n\t"
    "add     x1, x1, x3\n\t"
    "ldr     q31, [x1, #16*0]\n\t"
    "ldr     q26, [x0, #16*2]\n\t"
    "ldr     q27, [x0, #16*3]\n\t"
    "add     x0, x0, x2\n\t"
    "b       3b\n\t"

    //  Last iteration of the main loop (no loads beyond the panels)
    "4:\n\t"
    "fmla    v0.2d, v30.2d, v24.d[0]\n\t"
    "fmla    v3.2d, v30.2d, v24.d[1]\n\t"
    "fmla    v6.2d, v30.2d, v25.d[0]\n\t"
    "fmla    v9.2d, v30.2d, v25.d[1]\n\t"
    "fmla    v1.2d, v31.2d, v24.d[0]\n\t"
    "fmla    v4.2d, v31.2d, v24.d[1]\n\t"
    "fmla    v7.2d, v31.2d, v25.d[0]\n\t"
    "fmla    v10.2d, v31.2d, v25.d[1]\n\t"
    "fmla    v2.2d, v28.2d, v24.d[0]\n\t"
    "fmla    v5.2d, v28.2d, v24.d[1]\n\t"
    "fmla    v8.2d, v28.2d, v25.d[0]\n\t"
    "fmla    v11.2d, v28.2d, v25.d[1]\n\t"
    "ldr     q24, [x0, #0]\n\t"
    "ldr     q25, [x0, #0+16]\n\t"
    "fmla    v12.2d, v30.2d, v26.d[0]\n\t"
    "fmla    v15.2d, v30.2d, v26.d[1]\n\t"
    "fmla    v18.2d, v30.2d, v27.d[0]\n\t"
    "fmla    v21.2d, v30.2d, v27.d[1]\n\t"
    "ldr     q30, [x1, #16]\n\t"
    "fmla    v13.2d, v31.2d, v26.d[0]\n\t"
    "fmla    v16.2d, v31.2d, v26.d[1]\n\t"
    "fmla    v19.2d, v31.2d, v27.d[0]\n\t"
    "fmla    v22.2d, v31.2d, v27.d[1]\n\t"
    "ldr     q31, [x1, #16+16]\n\t"
    "fmla    v14.2d, v28.2d, v26.d[0]\n\t"
    "fmla    v17.2d, v28.2d, v26.d[1]\n\t"
    "fmla    v20.2d, v28.2d, v27.d[0]\n\t"
    "fmla    v23.2d, v28.2d, v27.d[1]\n\t"
    "add     x1, x1, x3\n\t"
    "ldr     q26, [x0, #16*2]\n\t"
    "ldr     q27, [x0, #16*3]\n\t"
    "add     x0, x0, x2\n\t"
    "fmla    v0.2d, v29.2d, v24.d[0]\n\t"
    "fmla    v3.2d, v29.2d, v24.d[1]\n\t"
    "fmla    v6.2d, v29.2d, v25.d[0]\n\t"
    "fmla    v9.2d, v29.2d, v25.d[1]\n\t"
    "fmla    v1.2d, v30.2d, v24.d[0]\n\t"
    "fmla    v4.2d, v30.2d, v24.d[1]\n\t"
    "fmla    v7.2d, v30.2d, v25.d[0]\n\t"
    "fmla    v10.2d, v30.2d, v25.d[1]\n\t"
    "fmla    v2.2d, v31.2d, v24.d[0]\n\t"
    "fmla    v5.2d, v31.2d, v24.d[1]\n\t"
    "fmla    v8.2d, v31.2d, v25.d[0]\n\t"
    "fmla    v11.2d, v31.2d, v25.d[1]\n\t"
    "fmla    v12.2d, v29.2d, v26.d[0]\n\t"
    "fmla    v15.2d, v29.2d, v26.d[1]\n\t"
    "fmla    v18.2d, v29.2d, v27.d[0]\n\t"
    "fmla    v21.2d, v29.2d, v27.d[1]\n\t"
    "fmla    v13.2d, v30.2d, v26.d[0]\n\t"
    "fmla    v16.2d, v30.2d, v26.d[1]\n\t"
    "fmla    v19.2d, v30.2d, v27.d[0]\n\t"
    "fmla    v22.2d, v30.2d, v27.d[1]\n\t"
    "fmla    v14.2d, v31.2d, v26.d[0]\n\t"
    "fmla    v17.2d, v31.2d, v26.d[1]\n\t"
    "fmla    v20.2d, v31.2d, v27.d[0]\n\t"
    "fmla    v23.2d, v31.2d, v27.d[1]\n\t"

    //  Remaining k % 4 rank-1 updates
    "5:\n\t"
    "cmp     x8, #0\n\t"
    "b.eq    6f\n\t"
    "ldr     q24, [x0, #16*0]\n\t"
    "ldr     q25, [x0, #16*1]\n\t"
    "ldr     q26, [x0, #16*2]\n\t"
    "ldr     q27, [x0, #16*3]\n\t"
    "add     x0, x0, x2\n\t"
    "ldr     q28, [x1, #16*0]\n\t"
    "ldr     q29, [x1, #16*1]\n\t"
    "ldr     q30, [x1, #16*2]\n\t"
    "add     x1, x1, x3\n\t"
    "sub     x8, x8, #1\n\t"
    "fmla    v0.2d, v28.2d, v24.d[0]\n\t"
    "fmla    v3.2d, v28.2d, v24.d[1]\n\t"
    "fmla    v6.2d, v28.2d, v25.d[0]\n\t"
    "fmla    v9.2d, v28.2d, v25.d[1]\n\t"
    "fmla    v1.2d, v29.2d, v24.d[0]\n\t"
    "fmla    v4.2d, v29.2d, v24.d[1]\n\t"
    "fmla    v7.2d, v29.2d, v25.d[0]\n\t"
    "fmla    v10.2d, v29.2d, v25.d[1]\n\t"
    "fmla    v2.2d, v30.2d, v24.d[0]\n\t"
    "fmla    v5.2d, v30.2d, v24.d[1]\n\t"
    "fmla    v8.2d, v30.2d, v25.d[0]\n\t"
    "fmla    v11.2d, v30.2d, v25.d[1]\n\t"
    "fmla    v12.2d, v28.2d, v26.d[0]\n\t"
    "fmla    v15.2d, v28.2d, v26.d[1]\n\t"
    "fmla    v18.2d, v28.2d, v27.d[0]\n\t"
    "fmla    v21.2d, v28.2d, v27.d[1]\n\t"
    "fmla    v13.2d, v29.2d, v26.d[0]\n\t"
    "fmla    v16.2d, v29.2d, v26.d[1]\n\t"
    "fmla    v19.2d, v29.2d, v27.d[0]\n\t"
    "fmla    v22.2d, v29.2d, v27.d[1]\n\t"
    "fmla    v14.2d, v30.2d, v26.d[0]\n\t"
    "fmla    v17.2d, v30.2d, v26.d[1]\n\t"
    "fmla    v20.2d, v30.2d, v27.d[0]\n\t"
    "fmla    v23.2d, v30.2d, v27.d[1]\n\t"
    "b       5b\n\t"

    //  alpha and beta, prefetch of the next panels, tile <- alpha*tile
    "6:\n\t"
    "ldr     x4, %[alpha]\n\t"
    "ldr     x8, %[beta]\n\t"
    "ld1r    {v24.2d}, [x4]\n\t"
    "ld1r    {v25.2d}, [x8]\n\t"
    "7:\n\t"
    "ldr     x0, %[a_next]\n\t"
    "ldr     x1, %[b_next]\n\t"
    "prfm    PLDL1STRM, [x0, 64*0]\n\t"
    "prfm    PLDL1STRM, [x0, 64*1]\n\t"
    "prfm    PLDL1STRM, [x0, 64*2]\n\t"
    "prfm    PLDL1STRM, [x1, 64*0]\n\t"
    "prfm    PLDL1STRM, [x1, 64*1]\n\t"
    "prfm    PLDL1STRM, [x1, 64*3]\n\t"
    "fmov    d26, #1.0\n\t"
    "fcmp    d24, d26\n\t"
    "b.eq    8f\n\t"
    "fmul    v0.2d, v0.2d, v24.d[0]\n\t"
    "fmul    v1.2d, v1.2d, v24.d[0]\n\t"
    "fmul    v2.2d, v2.2d, v24.d[0]\n\t"
    "fmul    v3.2d, v3.2d, v24.d[0]\n\t"
    "fmul    v4.2d, v4.2d, v24.d[0]\n\t"
    "fmul    v5.2d, v5.2d, v24.d[0]\n\t"
    "fmul    v6.2d, v6.2d, v24.d[0]\n\t"
    "fmul    v7.2d, v7.2d, v24.d[0]\n\t"
    "fmul    v8.2d, v8.2d, v24.d[0]\n\t"
    "fmul    v9.2d, v9.2d, v24.d[0]\n\t"
    "fmul    v10.2d, v10.2d, v24.d[0]\n\t"
    "fmul    v11.2d, v11.2d, v24.d[0]\n\t"
    "fmul    v12.2d, v12.2d, v24.d[0]\n\t"
    "fmul    v13.2d, v13.2d, v24.d[0]\n\t"
    "fmul    v14.2d, v14.2d, v24.d[0]\n\t"
    "fmul    v15.2d, v15.2d, v24.d[0]\n\t"
    "fmul    v16.2d, v16.2d, v24.d[0]\n\t"
    "fmul    v17.2d, v17.2d, v24.d[0]\n\t"
    "fmul    v18.2d, v18.2d, v24.d[0]\n\t"
    "fmul    v19.2d, v19.2d, v24.d[0]\n\t"
    "fmul    v20.2d, v20.2d, v24.d[0]\n\t"
    "fmul    v21.2d, v21.2d, v24.d[0]\n\t"
    "fmul    v22.2d, v22.2d, v24.d[0]\n\t"
    "fmul    v23.2d, v23.2d, v24.d[0]\n\t"

    //  C <- beta*C + tile (C is not read if beta == 0)
    "8:\n\t"
    "mov     x9, x5\n\t"
    "9:\n\t"
    "fcmp    d25, #0.0\n\t"
    "b.eq    10f\n\t"
    "ldr     q26, [x9, #0]\n\t"
    "ldr     q27, [x9, #0+16]\n\t"
    "ldr     q28, [x9, #0+32]\n\t"
    "add     x9, x9, x6\n\t"
    "ldr     q29, [x9, #0]\n\t"
    "ldr     q30, [x9, #0+16]\n\t"
    "ldr     q31, [x9, #0+32]\n\t"
    "add     x9, x9, x6\n\t"
    "fmla    v0.2d, v26.2d, v25.d[0]\n\t"
    "fmla    v1.2d, v27.2d, v25.d[0]\n\t"
    "fmla    v2.2d, v28.2d, v25.d[0]\n\t"
    "fmla    v3.2d, v29.2d, v25.d[0]\n\t"
    "fmla    v4.2d, v30.2d, v25.d[0]\n\t"
    "fmla    v5.2d, v31.2d, v25.d[0]\n\t"
    "10:\n\t"
    "str     q0, [x5, #0]\n\t"
    "str     q1, [x5, #0+16]\n\t"
    "str     q2, [x5, #0+32]\n\t"
    "add     x5, x5, x6\n\t"
    "str     q3, [x5, #0]\n\t"
    "str     q4, [x5, #0+16]\n\t"
    "str     q5, [x5, #0+32]\n\t"
    "add     x5, x5, x6\n\t"
    "b.eq    11f\n\t"
    "ldr     q26, [x9, #0]\n\t"
    "ldr     q27, [x9, #0+16]\n\t"
    "ldr     q28, [x9, #0+32]\n\t"
    "add     x9, x9, x6\n\t"
    "ldr     q29, [x9, #0]\n\t"
    "ldr     q30, [x9, #0+16]\n\t"
    "ldr     q31, [x9, #0+32]\n\t"
    "add     x9, x9, x6\n\t"
    "ldr     q0, [x9, #0]\n\t"
    "ldr     q1, [x9, #0+16]\n\t"
    "ldr     q2, [x9, #0+32]\n\t"
    "add     x9, x9, x6\n\t"
    "ldr     q3, [x9, #0]\n\t"
    "ldr     q4, [x9, #0+16]\n\t"
    "ldr     q5, [x9, #0+32]\n\t"
    "add     x9, x9, x6\n\t"
    "fmla    v6.2d, v26.2d, v25.d[0]\n\t"
    "fmla    v7.2d, v27.2d, v25.d[0]\n\t"
    "fmla    v8.2d, v28.2d, v25.d[0]\n\t"
    "fmla    v9.2d, v29.2d, v25.d[0]\n\t"
    "fmla    v10.2d, v30.2d, v25.d[0]\n\t"
    "fmla    v11.2d, v31.2d, v25.d[0]\n\t"
    "fmla    v12.2d, v0.2d, v25.d[0]\n\t"
    "fmla    v13.2d, v1.2d, v25.d[0]\n\t"
    "fmla    v14.2d, v2.2d, v25.d[0]\n\t"
    "fmla    v15.2d, v3.2d, v25.d[0]\n\t"
    "fmla    v16.2d, v4.2d, v25.d[0]\n\t"
    "fmla    v17.2d, v5.2d, v25.d[0]\n\t"
    "11:\n\t"
    "str     q6, [x5, #0]\n\t"
    "str     q7, [x5, #0+16]\n\t"
    "str     q8, [x5, #0+32]\n\t"
    "add     x5, x5, x6\n\t"
    "str     q9, [x5, #0]\n\t"
    "str     q10, [x5, #0+16]\n\t"
    "str     q11, [x5, #0+32]\n\t"
    "add     x5, x5, x6\n\t"
    "str     q12, [x5, #0]\n\t"
    "str     q13, [x5, #0+16]\n\t"
    "str     q14, [x5, #0+32]\n\t"
    "add     x5, x5, x6\n\t"
    "str     q15, [x5, #0]\n\t"
    "str     q16, [x5, #0+16]\n\t"
    "str     q17, [x5, #0+32]\n\t"
    "add     x5, x5, x6\n\t"
    "b.eq    12f\n\t"
    "ldr     q26, [x9, #0]\n\t"
    "ldr     q27, [x9, #0+16]\n\t"
    "ldr     q28, [x9, #0+32]\n\t"
    "add     x9, x9, x6\n\t"
    "ldr     q29, [x9, #0]\n\t"
    "ldr     q30, [x9, #0+16]\n\t"
    "ldr     q31, [x9, #0+32]\n\t"
    "add     x9, x9, x6\n\t"
    "fmla    v18.2d, v26.2d, v25.d[0]\n\t"
    "fmla    v19.2d, v27.2d, v25.d[0]\n\t"
    "fmla    v20.2d, v28.2d, v25.d[0]\n\t"
    "fmla    v21.2d, v29.2d, v25.d[0]\n\t"
    "fmla    v22.2d, v30.2d, v25.d[0]\n\t"
    "fmla    v23.2d, v31.2d, v25.d[0]\n\t"
    "12:\n\t"
    "str     q18, [x5, #0]\n\t"
    "str     q19, [x5, #0+16]\n\t"
    "str     q20, [x5, #0+32]\n\t"
    "add     x5, x5, x6\n\t"
    "str     q21, [x5, #0]\n\t"
    "str     q22, [x5, #0+16]\n\t"
    "str     q23, [x5, #0+32]\n\t"
    "add     x5, x5, x6\n\t"
    "13:\n\t"
    :
    :   [a]      "m"(a),
        [b]      "m"(b),
        [c]      "m"(C),
        [rs_c]   "m"(rs_c),
        [k_mker] "m"(k_mker),
        [k_left] "m"(k_left),
        [alpha]  "m"(alpha_),
        [beta]   "m"(beta_),
        [a_next] "m"(a_next),
        [b_next] "m"(b_next)
    :   "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x8", "x9",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        "cc", "memory"
    );
}

} } // namespace neon, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_NEON_UGEMM_TCC
