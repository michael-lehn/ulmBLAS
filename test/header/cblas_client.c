/*
 * A client in the style of the CBLAS standard (reference CBLAS of LAPACK,
 * OpenBLAS, MKL), compiled with the cblas.h of ulmBLAS as C with -Wall
 * -Werror (test T0503, issue #0028): type names without "enum",
 * CBLAS_LAYOUT and CBLAS_ORDER, CBLAS_INDEX, C99 complex numbers passed as
 * void pointers, the real arguments of the complex routines (c of crotg,
 * alpha of cher, csscal) as real numbers.
 */
#include <complex.h>
#include <cblas.h>

int
main(void)
{
    float complex   ca[4] = { 1, 2, 3, 4 }, cx[2] = { 1, I }, cs;
    double complex  za[4] = { 1, 2, 3, 4 }, zx[2] = { 1, I }, zy[2];
    double          d[4]  = { 1, 2, 3, 4 };
    double complex  zalpha = 1, zbeta = 0, zdot;
    float complex   calpha = 1;
    float           c;
    double          z = 0;

    CBLAS_LAYOUT    layout = CblasColMajor;
    CBLAS_ORDER     order  = CblasRowMajor;
    CBLAS_TRANSPOSE trans  = CblasConjTrans;
    CBLAS_UPLO      uplo   = CblasUpper;
    CBLAS_DIAG      diag   = CblasNonUnit;
    CBLAS_SIDE      side   = CblasLeft;
    CBLAS_INDEX     i;

    cblas_zgemv(layout, trans, 2, 2, &zalpha, za, 2, zx, 1, &zbeta, zy, 1);
    cblas_zhemv(order, uplo, 2, &zalpha, za, 2, zx, 1, &zbeta, zy, 1);
    cblas_zgemm(layout, trans, CblasNoTrans, 2, 2, 2, &zalpha, za, 2, za, 2,
                &zbeta, za, 2);
    cblas_ztrsm(layout, side, uplo, CblasNoTrans, diag, 2, 2, &zalpha, za, 2,
                za, 2);
    cblas_zherk(layout, uplo, CblasNoTrans, 2, 2, 1.0, za, 2, 0.0, za, 2);
    cblas_zher2k(layout, uplo, CblasNoTrans, 2, 2, &zalpha, za, 2, za, 2, 0.0,
                 za, 2);
    cblas_cher(layout, uplo, 2, 1.0f, cx, 1, ca, 2);
    cblas_csscal(2, 2.0f, cx, 1);
    cblas_zdotc_sub(2, zx, 1, zy, 1, &zdot);
    cblas_crotg(&ca[0], &ca[1], &c, &cs);
    cblas_caxpy(2, &calpha, cx, 1, cx, 1);
    i = cblas_izamax(2, zx, 1);
    z += cblas_dznrm2(2, zx, 1) + cblas_dcabs1(&zx[0]) + cblas_scasum(2, cx, 1);
    cblas_dgemm(layout, CblasNoTrans, CblasTrans, 2, 2, 2, 1.0, d, 2, d, 2,
                0.0, d, 2);
    return (int) i + (z < 0) + (c < 0);
}
