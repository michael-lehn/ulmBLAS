#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_ZGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_ZGEMM_TCC 1

//
//  ulmBLAS: experimental SME2 micro kernel for complex double (MR = NR =
//  16), written for ulmBLAS (not derived from another library).
//
//  The real and the imaginary part of the 16 x 16 tile of C are held in
//  ZA: ZA0.D to ZA3.D the real parts, ZA4.D to ZA7.D the imaginary parts,
//  tile 2*jb+ib (and 4+2*jb+ib) for the rows 8*ib ... 8*ib+7 and the
//  columns 8*jb ... 8*jb+7 of the tile of C.  As in the DGEMM kernel
//  (ugemm.tcc) the operands of the outer products are swapped, so that a
//  horizontal slice of a tile is a part of a column of C.
//
//  The packed panels are those of ulmBLAS for std::complex<double> (per k
//  16 consecutive complex elements of A and of B, real and imaginary part
//  interleaved): LD2D loads 8 complex elements and separates them into a
//  vector of real parts and one of imaginary parts.  Per k: 4 LD2D, and
//  for each of the four 8 x 8 blocks of the tile
//
//      Re += re(b) re(a)^T - im(b) im(a)^T     (FMOPA, FMOPS)
//      Im += im(b) re(a)^T + re(b) im(a)^T     (FMOPA, FMOPA)
//
//  i.e. 16 outer products into 8 tiles.  At the end the slices are read,
//  multiplied by alpha, added to beta*C (C is not read for beta = 0) and
//  stored with ST2D (interleaving real and imaginary parts again).  C with
//  unit row stride; partial tiles (mr, nr < 16) with predicated loads and
//  stores of the rows and the number of columns.  NaN in alpha, beta, A, B
//  or C propagates.
//
//  Requirements as for the DGEMM kernel: SME2, FEAT_SME_F64F64, SVL = 512
//  bits (Apple M4, M5), USE_SME.
//
#include <arm_sme.h>
#include <complex>
#include <cstdint>

namespace ulmBLAS { namespace sme {

//
//  The columns c0 ... c0+n-1 (n <= 8) of the tile of C from the tiles
//  T, T+1 (real parts) and T+4, T+5 (imaginary parts): C <- alpha*AB +
//  beta*C for the rows selected by p0 (rows 0..7) and p1 (rows 8..15).
//
template <unsigned T>
inline void
zupdate_16x8(long n, double ar, double ai, double br, double bi, bool beta0,
             double *C, long ldC, svbool_t p0, svbool_t p1)
    __arm_streaming __arm_in("za")
{
    const svbool_t    p  = svptrue_b64();
    const svfloat64_t z  = svdup_f64(0.0);
    const svfloat64_t var = svdup_f64(ar), vai = svdup_f64(ai);
    const svfloat64_t vbr = svdup_f64(br), vbi = svdup_f64(bi);

    for (long c=0; c<n; ++c) {
        double *Cc = C + 2*c*ldC;

        svfloat64_t re0 = svread_hor_za64_f64_m(z, p, T+0, c);
        svfloat64_t im0 = svread_hor_za64_f64_m(z, p, T+4, c);
        svfloat64_t re1 = svread_hor_za64_f64_m(z, p, T+1, c);
        svfloat64_t im1 = svread_hor_za64_f64_m(z, p, T+5, c);

        //  alpha * (re + i im)
        svfloat64_t or0 = svmls_x(p, svmul_x(p, re0, var), im0, vai);
        svfloat64_t oi0 = svmla_x(p, svmul_x(p, im0, var), re0, vai);
        svfloat64_t or1 = svmls_x(p, svmul_x(p, re1, var), im1, vai);
        svfloat64_t oi1 = svmla_x(p, svmul_x(p, im1, var), re1, vai);

        if (!beta0) {
            svfloat64x2_t c0 = svld2_f64(p0, Cc);
            svfloat64x2_t c1 = svld2_f64(p1, Cc + 16);
            svfloat64_t   cr0 = svget2(c0, 0), ci0 = svget2(c0, 1);
            svfloat64_t   cr1 = svget2(c1, 0), ci1 = svget2(c1, 1);

            or0 = svmls_x(p, svmla_x(p, or0, cr0, vbr), ci0, vbi);
            oi0 = svmla_x(p, svmla_x(p, oi0, ci0, vbr), cr0, vbi);
            or1 = svmls_x(p, svmla_x(p, or1, cr1, vbr), ci1, vbi);
            oi1 = svmla_x(p, svmla_x(p, oi1, ci1, vbr), cr1, vbi);
        }
        svst2_f64(p0, Cc, svcreate2(or0, oi0));
        svst2_f64(p1, Cc + 16, svcreate2(or1, oi1));
    }
}

//
//  The rows 0 ... mr-1 and columns 0 ... nr-1 of the 16 x 16 tile of C
//  (unit row stride, column stride ldC in complex elements), in streaming
//  mode.  A, B, C as arrays of double (real and imaginary parts).
//
__arm_new("za") __arm_locally_streaming
inline void
zgemm_16x16_streaming(long kc, double ar, double ai, const double *A,
                      const double *B, double br, double bi, double *C,
                      long ldC, long mr, long nr)
{
    const svbool_t p  = svptrue_b64();
    const svbool_t p0 = svwhilelt_b64_s64(int64_t(0), int64_t(mr));
    const svbool_t p1 = svwhilelt_b64_s64(int64_t(8), int64_t(mr));
    const long     n0 = (nr<8) ? nr : 8;
    const long     n1 = nr - n0;

    svzero_za();
    for (long l=0; l<kc; ++l) {
        svfloat64x2_t a0 = svld2_f64(p, A);
        svfloat64x2_t a1 = svld2_f64(p, A + 16);
        svfloat64x2_t b0 = svld2_f64(p, B);
        svfloat64x2_t b1 = svld2_f64(p, B + 16);

        svfloat64_t ar0 = svget2(a0, 0), ai0 = svget2(a0, 1);
        svfloat64_t ar1 = svget2(a1, 0), ai1 = svget2(a1, 1);
        svfloat64_t br0 = svget2(b0, 0), bi0 = svget2(b0, 1);
        svfloat64_t br1 = svget2(b1, 0), bi1 = svget2(b1, 1);

        //  columns 0..7 (jb = 0)
        svmopa_za64_f64_m(0, p, p, br0, ar0);
        svmopa_za64_f64_m(1, p, p, br0, ar1);
        svmopa_za64_f64_m(4, p, p, bi0, ar0);
        svmopa_za64_f64_m(5, p, p, bi0, ar1);
        //  columns 8..15 (jb = 1)
        svmopa_za64_f64_m(2, p, p, br1, ar0);
        svmopa_za64_f64_m(3, p, p, br1, ar1);
        svmopa_za64_f64_m(6, p, p, bi1, ar0);
        svmopa_za64_f64_m(7, p, p, bi1, ar1);

        svmops_za64_f64_m(0, p, p, bi0, ai0);
        svmops_za64_f64_m(1, p, p, bi0, ai1);
        svmopa_za64_f64_m(4, p, p, br0, ai0);
        svmopa_za64_f64_m(5, p, p, br0, ai1);
        svmops_za64_f64_m(2, p, p, bi1, ai0);
        svmops_za64_f64_m(3, p, p, bi1, ai1);
        svmopa_za64_f64_m(6, p, p, br1, ai0);
        svmopa_za64_f64_m(7, p, p, br1, ai1);

        A += 32;
        B += 32;
    }

    const bool beta0 = br==0.0 && bi==0.0;
    zupdate_16x8<0>(n0, ar, ai, br, bi, beta0, C, ldC, p0, p1);
    zupdate_16x8<2>(n1, ar, ai, br, bi, beta0, C + 2*8*ldC, ldC, p0, p1);
}

//
//  The same, called from ordinary code (see ugemm_32x16 for the
//  prefetching of the tile of C below: 16 rows of complex double are 256
//  bytes, as the 32 rows of double).
//
inline void
zgemm_16x16(long kc, const std::complex<double> &alpha,
            const std::complex<double> *A, const std::complex<double> *B,
            const std::complex<double> &beta, std::complex<double> *C,
            long ldC, long mr, long nr)
{
    double *C_ = reinterpret_cast<double *>(C);
    for (long c=0; c<16; ++c) {
        __builtin_prefetch(C_ + 2*c*ldC + 32, 1, 2);
        __builtin_prefetch(C_ + 2*c*ldC + 48, 1, 2);
        __builtin_prefetch(C_ + 2*c*ldC + 63, 1, 2);
    }
    zgemm_16x16_streaming(kc, alpha.real(), alpha.imag(),
                          reinterpret_cast<const double *>(A),
                          reinterpret_cast<const double *>(B),
                          beta.real(), beta.imag(), C_, ldC, mr, nr);
}

//  C <- beta*C + C_ for an mr x nr tile with any strides (C not read for
//  beta = 0), C_ column major with leading dimension 16
template <typename IndexType>
void
zupdate_strided(IndexType mr, IndexType nr, const std::complex<double> &beta,
                const std::complex<double> *C_, std::complex<double> *C,
                IndexType incRowC, IndexType incColC)
{
    const bool beta0 = beta.real()==0.0 && beta.imag()==0.0;
    for (IndexType j=0; j<nr; ++j) {
        for (IndexType i=0; i<mr; ++i) {
            std::complex<double> &c = C[i*incRowC+j*incColC];
            c = beta0 ? C_[i+16*j] : beta*c + C_[i+16*j];
        }
    }
}

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
void>::type
ugemm(IndexType                     kc,
      const std::complex<double>    &alpha,
      const std::complex<double>    *A,
      const std::complex<double>    *B,
      const std::complex<double>    &beta,
      std::complex<double>          *C,
      IndexType                     incRowC,
      IndexType                     incColC,
      const std::complex<double>    *,
      const std::complex<double>    *)
{
    if (incRowC==IndexType(1)) {
        zgemm_16x16(long(kc), alpha, A, B, beta, C, long(incColC), 16, 16);
        return;
    }
    std::complex<double> C_[16*16];
    zgemm_16x16(long(kc), alpha, A, B, std::complex<double>(0), C_, 16, 16, 16);
    zupdate_strided(IndexType(16), IndexType(16), beta, C_, C, incRowC,
                    incColC);
}

} // namespace sme

//
//  Buffered variant for complex double with the SME2 kernel (see the one
//  for double in ugemm.h): partial tiles of C with unit row stride are
//  updated directly.
//
template <typename IndexType>
void
ugemm(IndexType                     mr,
      IndexType                     nr,
      IndexType                     kc,
      const std::complex<double>    &alpha,
      const std::complex<double>    *A,
      const std::complex<double>    *B,
      const std::complex<double>    &beta,
      std::complex<double>          *C,
      IndexType                     incRowC,
      IndexType                     incColC,
      const std::complex<double>    *,
      const std::complex<double>    *)
{
    if (incRowC==IndexType(1)) {
        sme::zgemm_16x16(long(kc), alpha, A, B, beta, C, long(incColC),
                         long(mr), long(nr));
        return;
    }
    std::complex<double> C_[16*16];
    sme::zgemm_16x16(long(kc), alpha, A, B, std::complex<double>(0), C_, 16,
                     16, 16);
    sme::zupdate_strided(mr, nr, beta, C_, C, incRowC, incColC);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_ZGEMM_TCC
