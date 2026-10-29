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
//  ulmBLAS: experimental SME2 micro kernel for double (MR = 32, NR = 16).
//
//  Provenance: the structure of the kernel is that of the double precision
//  kernel bli_dgemm_armsme_int_4SVLx2SVL of BLIS pull request #921
//  (https://github.com/flame/blis/pull/921, configuration m4sme_p, file
//  kernels/armsme/3/bli_gemm_armsme_int_2SVLx2SVL.c, commit
//  235ef8ec097c43ab8299c89071af1b5b5e180843 of the fork Luislo1/blis):
//  the 32 x 16 tile of C is held in the eight 8 x 8 tiles ZA0.D to ZA7.D of
//  the ZA array (SVL = 512 bits), for each k one multi-vector load of the
//  column of the panel of A (4 vectors), one of the row of the panel of B
//  (2 vectors) and eight FMOPA (outer product and accumulate).
//
//  Modifications for ulmBLAS:
//
//  - Interface of the ulmBLAS micro kernels (alpha and beta by reference,
//    strides of C as IndexType), written in C++ with the ACLE intrinsics.
//  - The tiles hold the transposed blocks of C (the operands of FMOPA are
//    swapped): tile ZA(i+4*j) holds the columns 8*j to 8*j+7 of the rows
//    8*i to 8*i+7 of the tile of C in its horizontal slices, so that a
//    horizontal slice is a part of a column of C (BLIS: vertical slices;
//    horizontal slices were 3 to 5% faster for kc = 64 on an Apple M5, the
//    same for kc >= 1024).
//  - For alpha == 1 and beta == 0 or beta == 1 (the calls of the frame
//    algorithm of gemm for alpha = 1), the tile of C is loaded into ZA (beta
//    == 1) or ZA is zeroed (beta == 0) before the loop over k, and ZA is
//    stored directly into C (LD1D and ST1D of slices of ZA).  For other
//    alpha and beta the slices are read into vector registers, scaled and
//    added to beta*C (as BLIS; beta == 0 does not read C).
//  - C with unit row stride only; for other strides (BLIS: the temporary
//    tile) alpha*A*B is computed in a local column major tile and C is
//    updated in C++.
//  - Partial tiles (mr < 32 or nr < 16, the buffered variant ugemm(mr, nr,
//    kc, ...) of the macro kernels) are updated directly: rows with
//    predicates, columns by the number of slices (BLIS: the temporary
//    tile).
//
//  NaN is neither 1 nor 0, so alpha = NaN and beta = NaN are computed as
//  for any other value.  nextA and nextB are not used; the tile of C below
//  is prefetched (see ugemm_32x16).
//
//  Requirements: SME2 with FEAT_SME_F64F64 and SVL = 512 bits (Apple M4,
//  M5), selected with USE_SME (see ulmblas/config/simd.h).  The kernel
//  (ugemm_32x16_streaming) is executed in streaming mode with new ZA state
//  (__arm_locally_streaming, __arm_new("za")); ugemm_32x16 prefetches in
//  ordinary mode and calls it, so it can be called from ordinary code.  No alignment
//  of the panels is required.  A and B are read exactly within the panels
//  (MR*kc and NR*kc elements).
//

#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_UGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_UGEMM_TCC 1

#include <arm_sme.h>
#include <cstdint>
#include <ulmblas/level3/ukernel/sme/ugemm.h>

namespace ulmBLAS { namespace sme {

//
//  Columns 0 to n-1 (n <= 8) of the tiles T to T+3 (the rows 0 to 31 of 8
//  columns of the tile of C, column c at C + c*ldC).  p0 to p3 are the rows
//  of the tiles T to T+3 that are part of C (mr < 32 for a partial tile),
//  pc the same for the 32 rows as predicate-as-counter.  Inactive rows are
//  neither read nor written.
//
template <unsigned T>
inline void
load_32x8(long n, const double *C, long ldC, svbool_t p0, svbool_t p1,
          svbool_t p2, svbool_t p3)
    __arm_streaming __arm_inout("za")
{
    for (long c=0; c<n; ++c) {
        svld1_hor_za64(T+0, c, p0, C + c*ldC);
        svld1_hor_za64(T+1, c, p1, C + c*ldC + 8);
        svld1_hor_za64(T+2, c, p2, C + c*ldC + 16);
        svld1_hor_za64(T+3, c, p3, C + c*ldC + 24);
    }
}

template <unsigned T>
inline void
store_32x8(long n, double *C, long ldC, svbool_t p0, svbool_t p1,
           svbool_t p2, svbool_t p3)
    __arm_streaming __arm_in("za")
{
    for (long c=0; c<n; ++c) {
        svst1_hor_za64(T+0, c, p0, C + c*ldC);
        svst1_hor_za64(T+1, c, p1, C + c*ldC + 8);
        svst1_hor_za64(T+2, c, p2, C + c*ldC + 16);
        svst1_hor_za64(T+3, c, p3, C + c*ldC + 24);
    }
}

template <unsigned T>
inline void
update_32x8(long n, double alpha, double beta, double *C, long ldC,
            svcount_t pc)
    __arm_streaming __arm_in("za")
{
    const svbool_t    p  = svptrue_b64();
    const svfloat64_t va = svdup_f64(alpha);
    const svfloat64_t vb = svdup_f64(beta);

    for (long c=0; c<n; ++c) {
        double *Cc = C + c*ldC;

        svfloat64_t z0 = svmul_x(p, svread_hor_za64_f64_m(va, p, T+0, c), va);
        svfloat64_t z1 = svmul_x(p, svread_hor_za64_f64_m(va, p, T+1, c), va);
        svfloat64_t z2 = svmul_x(p, svread_hor_za64_f64_m(va, p, T+2, c), va);
        svfloat64_t z3 = svmul_x(p, svread_hor_za64_f64_m(va, p, T+3, c), va);

        if (beta!=0.0) {
            svfloat64x4_t x = svld1_f64_x4(pc, Cc);

            z0 = svmla_x(p, z0, svget4(x, 0), vb);
            z1 = svmla_x(p, z1, svget4(x, 1), vb);
            z2 = svmla_x(p, z2, svget4(x, 2), vb);
            z3 = svmla_x(p, z3, svget4(x, 3), vb);
        }
        svst1_f64_x4(pc, Cc, svcreate4(z0, z1, z2, z3));
    }
}

//
//  The rows 0 to mr-1 and columns 0 to nr-1 of the 32 x 16 tile of C (mr
//  <= 32, nr <= 16) with unit row stride and column stride ldC, in streaming
//  mode (see ugemm_32x16).
//
__arm_new("za") __arm_locally_streaming
inline void
ugemm_32x16_streaming(long kc, double alpha, const double *A,
                      const double *B, double beta, double *C, long ldC,
                      long mr, long nr)
{
    const svbool_t  p  = svptrue_b64();
    const svcount_t pc = svptrue_c64();
    const bool      direct = alpha==1.0 && (beta==0.0 || beta==1.0);

    const int64_t   m     = mr;
    const svbool_t  p0    = svwhilelt_b64_s64(int64_t(0), m);
    const svbool_t  p1    = svwhilelt_b64_s64(int64_t(8), m);
    const svbool_t  p2    = svwhilelt_b64_s64(int64_t(16), m);
    const svbool_t  p3    = svwhilelt_b64_s64(int64_t(24), m);
    const svcount_t pcr   = svwhilelt_c64_s64(int64_t(0), m, 4);
    const long      n0    = (nr<8) ? nr : 8;
    const long      n1    = nr - n0;

    svzero_za();
    if (direct && beta==1.0) {
        load_32x8<0>(n0, C, ldC, p0, p1, p2, p3);
        load_32x8<4>(n1, C + 8*ldC, ldC, p0, p1, p2, p3);
    }

    for (long l=0; l<kc; ++l) {
        svfloat64x4_t a = svld1_f64_x4(pc, A);
        svfloat64x2_t b = svld1_f64_x2(pc, B);

        svmopa_za64_f64_m(0, p, p, svget2(b, 0), svget4(a, 0));
        svmopa_za64_f64_m(1, p, p, svget2(b, 0), svget4(a, 1));
        svmopa_za64_f64_m(2, p, p, svget2(b, 0), svget4(a, 2));
        svmopa_za64_f64_m(3, p, p, svget2(b, 0), svget4(a, 3));
        svmopa_za64_f64_m(4, p, p, svget2(b, 1), svget4(a, 0));
        svmopa_za64_f64_m(5, p, p, svget2(b, 1), svget4(a, 1));
        svmopa_za64_f64_m(6, p, p, svget2(b, 1), svget4(a, 2));
        svmopa_za64_f64_m(7, p, p, svget2(b, 1), svget4(a, 3));

        A += 32;
        B += 16;
    }

    if (direct) {
        store_32x8<0>(n0, C, ldC, p0, p1, p2, p3);
        store_32x8<4>(n1, C + 8*ldC, ldC, p0, p1, p2, p3);
    } else {
        update_32x8<0>(n0, alpha, beta, C, ldC, pcr);
        update_32x8<4>(n1, alpha, beta, C + 8*ldC, ldC, pcr);
    }
}

//
//  The same, called from ordinary code.  Before entering streaming mode the
//  tile of C below is prefetched (into L2), the tile of the next call of
//  the macro kernel (mgemm): C comes from memory, and loading it into ZA
//  without prefetching cost up to 8% for n = 1024 on an Apple M5.
//  Prefetching the own tile at the beginning was too late.  PRFM does not
//  fault, even for addresses outside of C.  The 48 PRFM cost about 45 to
//  85 ns per call in streaming mode, almost nothing in ordinary mode (the
//  fixed cost of a call, which matters for small kc, about 115 ns before).
//
inline void
ugemm_32x16(long kc, double alpha, const double *A, const double *B,
            double beta, double *C, long ldC, long mr, long nr)
{
    for (long c=0; c<16; ++c) {
        __builtin_prefetch(C + c*ldC + 32, 1, 2);
        __builtin_prefetch(C + c*ldC + 48, 1, 2);
        __builtin_prefetch(C + c*ldC + 63, 1, 2);
    }
    ugemm_32x16_streaming(kc, alpha, A, B, beta, C, ldC, mr, nr);
}

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
      const double   *,
      const double   *)
{
    constexpr IndexType MR = BlockSizeUGemm<double>::MR;
    constexpr IndexType NR = BlockSizeUGemm<double>::NR;

    if (incRowC!=IndexType(1)) {
        double C_[MR*NR];

        ugemm_32x16(kc, alpha, A, B, 0.0, C_, MR, MR, NR);
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
    ugemm_32x16(kc, alpha, A, B, beta, C, incColC, MR, NR);
}

} // namespace sme

template <typename IndexType>
void
ugemm(IndexType      mr,
      IndexType      nr,
      IndexType      kc,
      const double   &alpha,
      const double   *A,
      const double   *B,
      const double   &beta,
      double         *C,
      IndexType      incRowC,
      IndexType      incColC,
      const double   *,
      const double   *)
{
    constexpr IndexType MR = sme::BlockSizeUGemm<double>::MR;
    constexpr IndexType NR = sme::BlockSizeUGemm<double>::NR;

    if (incRowC==IndexType(1)) {
        sme::ugemm_32x16(kc, alpha, A, B, beta, C, incColC, mr, nr);
        return;
    }

    double C_[MR*NR];

    sme::ugemm_32x16(kc, alpha, A, B, 0.0, C_, MR, MR, NR);
    if (beta==0.0) {
        for (IndexType j=0; j<nr; ++j) {
            for (IndexType i=0; i<mr; ++i) {
                C[i*incRowC+j*incColC] = C_[i+j*MR];
            }
        }
    } else {
        for (IndexType j=0; j<nr; ++j) {
            for (IndexType i=0; i<mr; ++i) {
                C[i*incRowC+j*incColC] = beta*C[i*incRowC+j*incColC]
                                       + C_[i+j*MR];
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_UGEMM_TCC
