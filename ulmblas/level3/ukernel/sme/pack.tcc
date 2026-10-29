#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_PACK_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_PACK_TCC 1

#include <arm_sme.h>
#include <cstdint>
#include <ulmblas/level3/ukernel/sme/pack.h>
#include <ulmblas/level3/ukernel/sme/ugemm.h>

namespace ulmBLAS {

//
//  With the SME2 kernel (eight times the flops of the NEON kernel per
//  cycle) the packing is a larger part of the time of gemm, in particular
//  for small m or n.  The panels are packed in streaming mode: loads and
//  stores of the streaming mode (executed by the SME unit) had a much
//  higher bandwidth on an Apple M5 than those of the ordinary mode.
//  Measured for blocks of 608 x 2240 (A) and 2240 x 3072 (B) of matrices
//  with 4096 rows (bytes read and written):
//
//  - A with unit row stride: copy of the 32 elements of each column with
//    multi-vector loads and stores (LD1D, ST1D of 4 vectors), 115 GB/s;
//    the portable routine 34 to 44 GB/s, with NEON and prefetching 41 GB/s
//    (prefetching slowed down the streaming mode to 70 GB/s).
//  - B with unit row stride (16 columns, each k is a row): ZA as transpose
//    buffer, 8 values of a column loaded into a horizontal slice of a tile
//    (LD1D), the vertical slices are 8 values of a row (ST1D), 72 GB/s; the
//    portable routine 45 to 52 GB/s, NEON transposes 53 GB/s.
//
//  The transposed cases (A with unit column stride, B with unit column
//  stride) are packed the same way with the roles of rows and columns
//  swapped.  Other strides use the portable routines.  For double conj has
//  no effect.
//

namespace sme {

//
//  Panel of 32 rows of A with unit row stride (column j: A + j*incColA);
//  the rows mr to 31 (mr <= 32) are zero.
//
__arm_locally_streaming
inline void
pack_A_colmajor(long mr, long k, const double *A, long incColA,
                double *buffer)
{
    const svcount_t pc = svptrue_c64();
    const svcount_t pm = svwhilelt_c64_s64(int64_t(0), int64_t(mr), 4);

    for (long j=0; j<k; ++j) {
        svst1_f64_x4(pc, buffer, svld1_f64_x4(pm, A));
        buffer += 32;
        A      += incColA;
    }
}

//
//  Panel of 32 rows of A with unit column stride (row i: A + i*incRowA);
//  the rows mr to 31 (mr <= 32) are zero.  Per 8 columns, row 8*t+c into
//  the horizontal slice c of tile t, the vertical slice r of the tiles 0 to
//  3 is column r of the panel (the slices of the rows mr to 31 stay zero).
//
__arm_new("za") __arm_locally_streaming
inline void
pack_A_rowmajor(long mr, long k, const double *A, long incRowA,
                double *buffer)
{
    const svbool_t p = svptrue_b64();
    long           j = 0;

    svzero_za();
    for (; j+8<=k; j+=8) {
        for (long c=0; c<8; ++c) {
            if (c   <mr) svld1_hor_za64(0, c, p, A + (c   )*incRowA + j);
            if (c+ 8<mr) svld1_hor_za64(1, c, p, A + (c+ 8)*incRowA + j);
            if (c+16<mr) svld1_hor_za64(2, c, p, A + (c+16)*incRowA + j);
            if (c+24<mr) svld1_hor_za64(3, c, p, A + (c+24)*incRowA + j);
        }
        for (long r=0; r<8; ++r) {
            svst1_ver_za64(0, r, p, buffer + 32*r);
            svst1_ver_za64(1, r, p, buffer + 32*r + 8);
            svst1_ver_za64(2, r, p, buffer + 32*r + 16);
            svst1_ver_za64(3, r, p, buffer + 32*r + 24);
        }
        buffer += 8*32;
    }
    //  The last k%8 columns the same way, with the loads predicated to the
    //  columns j to k-1 (inactive elements of a slice are set to zero).
    //  Scalar stores here (in streaming mode) made the following loads of
    //  the micro kernel slow: up to 1 us per panel on an Apple M5.
    if (j<k) {
        const svbool_t pk = svwhilelt_b64_s64(int64_t(j), int64_t(k));

        for (long c=0; c<8; ++c) {
            if (c   <mr) svld1_hor_za64(0, c, pk, A + (c   )*incRowA + j);
            if (c+ 8<mr) svld1_hor_za64(1, c, pk, A + (c+ 8)*incRowA + j);
            if (c+16<mr) svld1_hor_za64(2, c, pk, A + (c+16)*incRowA + j);
            if (c+24<mr) svld1_hor_za64(3, c, pk, A + (c+24)*incRowA + j);
        }
        for (long r=0; r<k-j; ++r) {
            svst1_ver_za64(0, r, p, buffer + 32*r);
            svst1_ver_za64(1, r, p, buffer + 32*r + 8);
            svst1_ver_za64(2, r, p, buffer + 32*r + 16);
            svst1_ver_za64(3, r, p, buffer + 32*r + 24);
        }
    }
}

//
//  Panel of 16 columns of B with unit row stride (column j: B + j*incColB);
//  the columns nr to 15 (nr <= 16) are zero.  Per 8 rows, column 8*t+c into
//  the horizontal slice c of tile t, the vertical slice r of the tiles 0 and
//  1 is row r of the panel (the slices of the columns nr to 15 stay zero).
//
__arm_new("za") __arm_locally_streaming
inline void
pack_B_colmajor(long nr, long k, const double *B, long incColB,
                double *buffer)
{
    const svbool_t p = svptrue_b64();
    long           l = 0;

    svzero_za();
    for (; l+8<=k; l+=8) {
        for (long c=0; c<8; ++c) {
            if (c  <nr) svld1_hor_za64(0, c, p, B + (c  )*incColB + l);
            if (c+8<nr) svld1_hor_za64(1, c, p, B + (c+8)*incColB + l);
        }
        for (long r=0; r<8; ++r) {
            svst1_ver_za64(0, r, p, buffer + 16*r);
            svst1_ver_za64(1, r, p, buffer + 16*r + 8);
        }
        buffer += 8*16;
    }
    //  The last k%8 rows the same way, with the loads predicated to the rows
    //  l to k-1 (see pack_A_rowmajor).
    if (l<k) {
        const svbool_t pk = svwhilelt_b64_s64(int64_t(l), int64_t(k));

        for (long c=0; c<8; ++c) {
            if (c  <nr) svld1_hor_za64(0, c, pk, B + (c  )*incColB + l);
            if (c+8<nr) svld1_hor_za64(1, c, pk, B + (c+8)*incColB + l);
        }
        for (long r=0; r<k-l; ++r) {
            svst1_ver_za64(0, r, p, buffer + 16*r);
            svst1_ver_za64(1, r, p, buffer + 16*r + 8);
        }
    }
}

//
//  Panel of 16 columns of B with unit column stride (row l: B + l*incRowB);
//  the columns nr to 15 (nr <= 16) are zero.
//
__arm_locally_streaming
inline void
pack_B_rowmajor(long nr, long k, const double *B, long incRowB,
                double *buffer)
{
    const svcount_t pc = svptrue_c64();
    const svcount_t pn = svwhilelt_c64_s64(int64_t(0), int64_t(nr), 2);

    for (long l=0; l<k; ++l) {
        svst1_f64_x2(pc, buffer, svld1_f64_x2(pn, B));
        buffer += 16;
        B      += incRowB;
    }
}

} // namespace sme

//
//  P consecutive panels of MR = 32 rows of A, panel by panel.
//
template <int P, typename IndexType>
static void
pack_MRxk(IndexType      k,
          bool           ,
          const double   *A,
          IndexType      incRowA,
          IndexType      incColA,
          double         *buffer)
{
    constexpr IndexType MR = sme::BlockSizeUGemm<double>::MR;

    static_assert(MR==32, "SME packing of A requires MR = 32");

    if (incRowA!=IndexType(1) && incColA!=IndexType(1)) {
        pack_MRxk<P, IndexType, double, double>(k, false, A, incRowA,
                                                incColA, buffer);
        return;
    }
    for (IndexType p=0; p<P; ++p) {
        if (incRowA==IndexType(1)) {
            sme::pack_A_colmajor(MR, k, A + p*MR, incColA, buffer + p*k*MR);
        } else {
            sme::pack_A_rowmajor(MR, k, A + p*MR*incRowA, incRowA,
                                 buffer + p*k*MR);
        }
    }
}

//
//  Panel of NR = 16 columns of B.
//
template <typename IndexType>
static void
pack_kxNR(IndexType      k,
          bool           ,
          const double   *B,
          IndexType      incRowB,
          IndexType      incColB,
          double         *buffer)
{
    constexpr IndexType NR = sme::BlockSizeUGemm<double>::NR;

    static_assert(NR==16, "SME packing of B requires NR = 16");

    if (incRowB==IndexType(1)) {
        sme::pack_B_colmajor(NR, k, B, incColB, buffer);
    } else if (incColB==IndexType(1)) {
        sme::pack_B_rowmajor(NR, k, B, incRowB, buffer);
    } else {
        pack_kxNR<IndexType, double, double>(k, false, B, incRowB, incColB,
                                             buffer);
    }
}

//
//  The last panel of A with mr < 32 rows (see gepack_A).
//
template <typename IndexType>
static bool
pack_MRxk_partial(IndexType      mr,
                  IndexType      k,
                  bool           ,
                  const double   *A,
                  IndexType      incRowA,
                  IndexType      incColA,
                  double         *buffer)
{
    if (incRowA==IndexType(1)) {
        sme::pack_A_colmajor(mr, k, A, incColA, buffer);
        return true;
    }
    if (incColA==IndexType(1)) {
        sme::pack_A_rowmajor(mr, k, A, incRowA, buffer);
        return true;
    }
    return false;
}

//
//  The last panel of B with nr < 16 columns (see gepack_B).
//
template <typename IndexType>
static bool
pack_kxNR_partial(IndexType      nr,
                  IndexType      k,
                  bool           ,
                  const double   *B,
                  IndexType      incRowB,
                  IndexType      incColB,
                  double         *buffer)
{
    if (incRowB==IndexType(1)) {
        sme::pack_B_colmajor(nr, k, B, incColB, buffer);
        return true;
    }
    if (incColB==IndexType(1)) {
        sme::pack_B_rowmajor(nr, k, B, incRowB, buffer);
        return true;
    }
    return false;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_PACK_TCC
