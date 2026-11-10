#ifndef ULMBLAS_STRIDED_H
#define ULMBLAS_STRIDED_H 1

/*
 *  GEMM with arbitrary row and column strides
 *
 *      C <- alpha op(A) op(B) + beta C
 *
 *  for float, double, complex float and complex double, and for the
 *  integer types int8_t, int16_t, int32_t and int64_t.  Every matrix is
 *  described by a pointer and two strides: element (i,j) of the m x n
 *  matrix X is
 *
 *      X[i*incRowX + j*incColX],   0 <= i < m, 0 <= j < n.
 *
 *  - Column major with leading dimension ld: incRow = 1, incCol = ld.  Row
 *    major: incRow = ld, incCol = 1.  A transposed matrix is passed by
 *    swapping its strides; there is no transposition argument.
 *  - Strides are counted in elements (a complex element counts as one),
 *    not in bytes.  They may be any value, also negative or zero: the
 *    pointer always points to element (0,0), and negative strides run
 *    backwards from there.
 *  - op(A) = A or conj(A) (conjA != 0), likewise for B: conjugation is
 *    the only operation besides the strides (complex types only).
 *  - Dimensions and strides are int64_t, independent of the index type of
 *    the BLAS interfaces (ULMBLAS_INDEX_TYPE).
 *  - Complex numbers: arrays of two values (real part, imaginary part),
 *    the layout of C99 float _Complex / double _Complex and of
 *    std::complex.  The scalars alpha and beta of the complex functions
 *    are passed by pointer to such a pair; the matrices as pointers to
 *    the real part of element (0,0).
 *
 *  Contract (not checked):
 *
 *  - m, n, k >= 0.
 *  - C must not overlap with A or B (no element of C may be stored where
 *    an element of A or B is read); otherwise the result is undefined.
 *  - The m x n elements of C must be distinct memory locations: a zero
 *    stride of C is allowed only for a dimension of length 1 (incRowC
 *    for m = 1, incColC for n = 1).
 *  - A and B are only read: their elements may overlap, so zero or
 *    otherwise "overlapping" strides are allowed for A and B (e.g. a row
 *    vector repeated as the rows of a matrix with incRow = 0).
 *
 *  Integers: the products and sums wrap around modulo 2^bits (two's
 *  complement), as Julia's Int arithmetic.  Use the same function for
 *  unsigned integers of the same size: the bits are the same.  There are no
 *  tuned micro kernels for integers yet; the generic one is used.
 *
 *  Semantics as in BLAS: for beta = 0, C is not read (NaN or Inf in C do
 *  not propagate); for alpha = 0 or k = 0, A and B are not read and C is
 *  only scaled by beta.  The functions are reentrant and use one thread.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void
ulmblas_sgemm_strided(int64_t m, int64_t n, int64_t k, float alpha,
                      const float *A, int64_t incRowA, int64_t incColA,
                      const float *B, int64_t incRowB, int64_t incColB,
                      float beta, float *C, int64_t incRowC, int64_t incColC);

void
ulmblas_dgemm_strided(int64_t m, int64_t n, int64_t k, double alpha,
                      const double *A, int64_t incRowA, int64_t incColA,
                      const double *B, int64_t incRowB, int64_t incColB,
                      double beta, double *C, int64_t incRowC, int64_t incColC);

void
ulmblas_cgemm_strided(int64_t m, int64_t n, int64_t k, const float *alpha,
                      int conjA, const float *A, int64_t incRowA,
                      int64_t incColA, int conjB, const float *B,
                      int64_t incRowB, int64_t incColB, const float *beta,
                      float *C, int64_t incRowC, int64_t incColC);

void
ulmblas_zgemm_strided(int64_t m, int64_t n, int64_t k, const double *alpha,
                      int conjA, const double *A, int64_t incRowA,
                      int64_t incColA, int conjB, const double *B,
                      int64_t incRowB, int64_t incColB, const double *beta,
                      double *C, int64_t incRowC, int64_t incColC);

void
ulmblas_i8gemm_strided(int64_t m, int64_t n, int64_t k, int8_t alpha,
                       const int8_t *A, int64_t incRowA, int64_t incColA,
                       const int8_t *B, int64_t incRowB, int64_t incColB,
                       int8_t beta, int8_t *C, int64_t incRowC,
                       int64_t incColC);

void
ulmblas_i16gemm_strided(int64_t m, int64_t n, int64_t k, int16_t alpha,
                        const int16_t *A, int64_t incRowA, int64_t incColA,
                        const int16_t *B, int64_t incRowB, int64_t incColB,
                        int16_t beta, int16_t *C, int64_t incRowC,
                        int64_t incColC);

void
ulmblas_i32gemm_strided(int64_t m, int64_t n, int64_t k, int32_t alpha,
                        const int32_t *A, int64_t incRowA, int64_t incColA,
                        const int32_t *B, int64_t incRowB, int64_t incColB,
                        int32_t beta, int32_t *C, int64_t incRowC,
                        int64_t incColC);

void
ulmblas_i64gemm_strided(int64_t m, int64_t n, int64_t k, int64_t alpha,
                        const int64_t *A, int64_t incRowA, int64_t incColA,
                        const int64_t *B, int64_t incRowB, int64_t incColB,
                        int64_t beta, int64_t *C, int64_t incRowC,
                        int64_t incColC);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* ULMBLAS_STRIDED_H */
