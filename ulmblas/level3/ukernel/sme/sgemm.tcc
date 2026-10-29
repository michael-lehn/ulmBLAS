#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_SGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_SGEMM_TCC 1

//
//  ulmBLAS: experimental SME2 micro kernel for float (MR = NR = 32),
//  written for ulmBLAS after the pattern of the DGEMM kernel (ugemm.tcc,
//  whose structure comes from BLIS pull request #921).  The tile layout,
//  four tiles for 2 SVL x 2 SVL elements, is also that of the single
//  precision kernel bli_sgemm_armsme_int_2SVLx2SVL of that pull request;
//  no code of it is used.
//
//  The 32 x 32 tile of C is held in the four 16 x 16 tiles ZA0.S to ZA3.S
//  of the ZA array (SVL = 512 bits): tile ib+2*jb for the rows 16*ib ...
//  16*ib+15 and the columns 16*jb ... 16*jb+15.  As in the DGEMM kernel
//  (ugemm.tcc) the operands of the outer products are swapped, so that a
//  horizontal slice of a tile is a part of a column of C.  Per k one
//  multi-vector load of the column of the panel of A (2 vectors), one of
//  the row of the panel of B (2 vectors) and four FMOPA.
//
//  As for double: for alpha == 1 and beta == 0 or beta == 1 the tile of C
//  is loaded into ZA (beta == 1) or ZA is zeroed (beta == 0) before the loop
//  over k and stored directly into C; otherwise the slices are read, scaled
//  and added to beta*C (beta == 0 does not read C).  C with unit row
//  stride; for other strides alpha*A*B is computed in a local tile and C is
//  updated in C++.  Partial tiles (mr, nr < 32) with predicated loads and
//  stores of the rows and the number of columns.  NaN is neither 1 nor 0,
//  so alpha = NaN and beta = NaN are computed as for any other value.
//
//  Requirements as for the DGEMM kernel: SME2, SVL = 512 bits (Apple M4,
//  M5), USE_SME.  The panels are packed by the routines of spack.tcc.
//
#include <arm_sme.h>
#include <cstdint>

namespace ulmBLAS { namespace sme {

//
//  Columns 0 to n-1 (n <= 16) of the tiles T and T+1 (the rows 0 to 31 of
//  16 columns of the tile of C, column c at C + c*ldC); p0 and p1 are the
//  rows of the tiles T and T+1 that are part of C.
//
template <unsigned T>
inline void
sload_32x16(long n, const float *C, long ldC, svbool_t p0, svbool_t p1)
    __arm_streaming __arm_inout("za")
{
    for (long c=0; c<n; ++c) {
        svld1_hor_za32(T+0, c, p0, C + c*ldC);
        svld1_hor_za32(T+1, c, p1, C + c*ldC + 16);
    }
}

template <unsigned T>
inline void
sstore_32x16(long n, float *C, long ldC, svbool_t p0, svbool_t p1)
    __arm_streaming __arm_in("za")
{
    for (long c=0; c<n; ++c) {
        svst1_hor_za32(T+0, c, p0, C + c*ldC);
        svst1_hor_za32(T+1, c, p1, C + c*ldC + 16);
    }
}

template <unsigned T>
inline void
supdate_32x16(long n, float alpha, float beta, float *C, long ldC,
              svcount_t pc)
    __arm_streaming __arm_in("za")
{
    const svbool_t    p  = svptrue_b32();
    const svfloat32_t va = svdup_f32(alpha);
    const svfloat32_t vb = svdup_f32(beta);

    for (long c=0; c<n; ++c) {
        float *Cc = C + c*ldC;

        svfloat32_t z0 = svmul_x(p, svread_hor_za32_f32_m(va, p, T+0, c), va);
        svfloat32_t z1 = svmul_x(p, svread_hor_za32_f32_m(va, p, T+1, c), va);

        if (beta!=0.0f) {
            svfloat32x2_t x = svld1_f32_x2(pc, Cc);

            z0 = svmla_x(p, z0, svget2(x, 0), vb);
            z1 = svmla_x(p, z1, svget2(x, 1), vb);
        }
        svst1_f32_x2(pc, Cc, svcreate2(z0, z1));
    }
}

//
//  The rows 0 to mr-1 and columns 0 to nr-1 of the 32 x 32 tile of C (mr,
//  nr <= 32) with unit row stride and column stride ldC, in streaming mode.
//
__arm_new("za") __arm_locally_streaming
inline void
sgemm_32x32_streaming(long kc, float alpha, const float *A, const float *B,
                      float beta, float *C, long ldC, long mr, long nr)
{
    const svbool_t  p  = svptrue_b32();
    const svcount_t pc = svptrue_c32();
    const bool      direct = alpha==1.0f && (beta==0.0f || beta==1.0f);

    const int64_t   m   = mr;
    const svbool_t  p0  = svwhilelt_b32_s64(int64_t(0), m);
    const svbool_t  p1  = svwhilelt_b32_s64(int64_t(16), m);
    const svcount_t pcr = svwhilelt_c32_s64(int64_t(0), m, 2);
    const long      n0  = (nr<16) ? nr : 16;
    const long      n1  = nr - n0;

    svzero_za();
    if (direct && beta==1.0f) {
        sload_32x16<0>(n0, C, ldC, p0, p1);
        sload_32x16<2>(n1, C + 16*ldC, ldC, p0, p1);
    }

    for (long l=0; l<kc; ++l) {
        svfloat32x2_t a = svld1_f32_x2(pc, A);
        svfloat32x2_t b = svld1_f32_x2(pc, B);

        svmopa_za32_f32_m(0, p, p, svget2(b, 0), svget2(a, 0));
        svmopa_za32_f32_m(1, p, p, svget2(b, 0), svget2(a, 1));
        svmopa_za32_f32_m(2, p, p, svget2(b, 1), svget2(a, 0));
        svmopa_za32_f32_m(3, p, p, svget2(b, 1), svget2(a, 1));

        A += 32;
        B += 32;
    }

    if (direct) {
        sstore_32x16<0>(n0, C, ldC, p0, p1);
        sstore_32x16<2>(n1, C + 16*ldC, ldC, p0, p1);
    } else {
        supdate_32x16<0>(n0, alpha, beta, C, ldC, pcr);
        supdate_32x16<2>(n1, alpha, beta, C + 16*ldC, ldC, pcr);
    }
}

//
//  The same, called from ordinary code: the tile of C below (the tile of
//  the next call of the macro kernel) is prefetched first, see ugemm_32x16.
//
inline void
sgemm_32x32(long kc, float alpha, const float *A, const float *B,
            float beta, float *C, long ldC, long mr, long nr)
{
    for (long c=0; c<32; ++c) {
        __builtin_prefetch(C + c*ldC + 32, 1, 2);
        __builtin_prefetch(C + c*ldC + 63, 1, 2);
    }
    sgemm_32x32_streaming(kc, alpha, A, B, beta, C, ldC, mr, nr);
}

//  C <- beta*C + C_ for an mr x nr tile with any strides (C not read for
//  beta = 0), C_ column major with leading dimension MR
template <int MR, typename IndexType, typename T>
void
update_strided(IndexType mr, IndexType nr, const T &beta, const T *C_, T *C,
               IndexType incRowC, IndexType incColC)
{
    const bool beta0 = beta==T(0);
    for (IndexType j=0; j<nr; ++j) {
        for (IndexType i=0; i<mr; ++i) {
            T &c = C[i*incRowC+j*incColC];
            c = beta0 ? C_[i+MR*j] : beta*c + C_[i+MR*j];
        }
    }
}

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
void>::type
ugemm(IndexType      kc,
      const float    &alpha,
      const float    *A,
      const float    *B,
      const float    &beta,
      float          *C,
      IndexType      incRowC,
      IndexType      incColC,
      const float    *,
      const float    *)
{
    if (incRowC==IndexType(1)) {
        sgemm_32x32(long(kc), alpha, A, B, beta, C, long(incColC), 32, 32);
        return;
    }
    float C_[32*32];
    sgemm_32x32(long(kc), alpha, A, B, 0.0f, C_, 32, 32, 32);
    update_strided<32>(IndexType(32), IndexType(32), beta, C_, C, incRowC,
                       incColC);
}

} // namespace sme

//
//  Buffered variant for float with the SME2 kernel (see the one for double
//  in ugemm.h): partial tiles of C with unit row stride are updated
//  directly.
//
template <typename IndexType>
void
ugemm(IndexType      mr,
      IndexType      nr,
      IndexType      kc,
      const float    &alpha,
      const float    *A,
      const float    *B,
      const float    &beta,
      float          *C,
      IndexType      incRowC,
      IndexType      incColC,
      const float    *,
      const float    *)
{
    if (incRowC==IndexType(1)) {
        sme::sgemm_32x32(long(kc), alpha, A, B, beta, C, long(incColC),
                         long(mr), long(nr));
        return;
    }
    float C_[32*32];
    sme::sgemm_32x32(long(kc), alpha, A, B, 0.0f, C_, 32, 32, 32);
    sme::update_strided<32>(mr, nr, beta, C_, C, incRowC, incColC);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_SGEMM_TCC
