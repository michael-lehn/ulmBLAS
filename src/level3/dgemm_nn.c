#include <ulmblas.h>
#include <stdio.h>
#include <emmintrin.h>

//
//  Macro kernel cache block sizes (i.e. panel widths and heights)
//

#define MC              384
#define KC              384
#define NC              4096

//
//  Micro kernel register block sizes
//
#define MR              4
#define NR              4

//
//  Macro kernel buffers for storing matrix panels in packed format
//

static double _A[MC*KC] __attribute__ ((aligned (16)));
static double _B[KC*NC] __attribute__ ((aligned (16)));
static double _C[MR*NR] __attribute__ ((aligned (16)));

void
ULMBLAS(pack_kx4)(long          m,
                  const double  * restrict A,
                  double        * restrict p,
                  long          panelSize)
{
    for (; m!=0; --m) {
        *(p+0) = *(A+0);
        *(p+1) = *(A+1);
        *(p+2) = *(A+2);
        *(p+3) = *(A+3);
        p+=panelSize;
        A += 4;
    }
}

void
ULMBLAS(pack_A_)(const long     m,
                const long     n,
                const double   * restrict A,
                const long     ldA)
{
    long j, panelSize;

    panelSize = 4*n;
    for (j=0; j<n; ++j) {
        ULMBLAS(pack_kx4)(m/4, A+ldA*j, _A+j*4, panelSize);
    }
    if (m%4 != 0) {
        fprintf(stderr, "(m%%4 != 0)\n");
    }
}

void
ULMBLAS(pack_4xk)(long          n,
                  const double  * restrict B,
                  const long    bRowInc,
                  const long    bColInc,
                  double        * restrict p)
{
    for (; n!=0; --n) {
        *(p+0) = *(B+0*bColInc);
        *(p+1) = *(B+1*bColInc);
        *(p+2) = *(B+2*bColInc);
        *(p+3) = *(B+3*bColInc);
        p+=4;
        B += bRowInc;
    }
}

//
//  Packing and padding panels from A into macro kernel buffer _A
//

void
ULMBLAS(packA_4xk)(long          n,                // %rdi
                   const double  * restrict A,     // %rsi
                   const long    ldA,              // %rdx
                   double        * restrict p)     // %rcx
{
    for (; n!=0; --n) {
        p[0] = A[0];
        p[1] = A[1];
        p[2] = A[2];
        p[3] = A[3];
        p+=4;
        A+=ldA;
    }
}

void
ULMBLAS(pack_A)(const long     m,
                const long     n,
                const double   * restrict A,
                const long     ldA)
{
//
//  m = M*MR + _MR with (0<=_MR<MR)
//
    const long M   = m / MR;
    const long _MR = m % MR;

    long         j, I;

    const double * restrict a  = A;
    double       * restrict _a = _A;

    for (I=0; I<M; ++I) {
        //ULMBLAS(packA_4xk)(n, a, ldA, _a);
        ULMBLAS(pack_4xk)(n, a, ldA, 1, _a);
        a  += MR;
        _a += MR*n;
    }
    if (_MR>0) {
        _a = &_A[n*M*MR];
        a = &A[M*MR];
        if (_MR==1) {
             for (j=0; j<n; ++j) {
                _a[0] = a[0];
                _a[1] = 0.0;
                _a[2] = 0.0;
                _a[3] = 0.0;
                _a += MR;
                a  += ldA;
             }
        } else if (_MR==2) {
             for (j=0; j<n; ++j) {
                _a[0] = a[0];
                _a[1] = a[1];
                _a[2] = 0.0;
                _a[3] = 0.0;
                _a += MR;
                a  += ldA;
             }
        } else if (_MR==3) {
             for (j=0; j<n; ++j) {
                _a[0] = a[0];
                _a[1] = a[1];
                _a[2] = a[2];
                _a[3] = 0.0;
                _a += MR;
                a  += ldA;
             }
        }
    }
}

//
//  Packing and padding panels from B into macro kernel buffer _B
//

void
ULMBLAS(packB_4xk)(long          m,
                   const double  * restrict B,
                   const long    ldB,
                   double        * restrict p)
{
    for (; m!=0; --m) {
        p[0] = B[0*ldB];
        p[1] = B[1*ldB];
        p[2] = B[2*ldB];
        p[3] = B[3*ldB];
        p+=4;
        ++B;
    }
}

void
ULMBLAS(pack_B)(const long     m,
                const long     n,
                const double   * restrict B,
                const long     ldB)
{
//
//  n = N*NR + _NR with (0<=_NR<NR)
//
    const long N   = n / NR;
    const long _NR = n % NR;

    long    i, J;

    const double * restrict b  = B;
    double       * restrict _b = _B;

    for (J=0; J<N; ++J) {
        //ULMBLAS(packB_4xk)(m, b, ldB, _b);
        ULMBLAS(pack_4xk)(m, b, 1, ldB, _b);
        b  += NR*ldB;
        _b += NR*m;
    }

    if (_NR>0) {
        b  = &B[N*NR*ldB];
        _b = &_B[NR*m*N];
        if (_NR==1) {
            for (i=0; i<m; ++i) {
                _b[0] = b[0*ldB];
                _b[1] = 0.0;
                _b[2] = 0.0;
                _b[3] = 0.0;
                _b += NR;
                ++b;
            }
        } else if (_NR==2) {
            for (i=0; i<m; ++i) {
                _b[0] = b[0*ldB];
                _b[1] = b[1*ldB];
                _b[2] = 0.0;
                _b[3] = 0.0;
                _b += NR;
                ++b;
            }
        } else if (_NR==3) {
            for (i=0; i<m; ++i) {
                _b[0] = b[0*ldB];
                _b[1] = b[1*ldB];
                _b[2] = b[2*ldB];
                _b[3] = 0.0;
                _b += NR;
                ++b;
            }
        }
    }
}

//
//  Unpack C from buffer _C
//

void
ULMBLAS(unpack_C)(const long    m,
                  const long    n,
                  const double  beta,
                  double        * restrict C,
                  const long    ldC)
{
    long i, j;

    if (beta!=0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                C[i+j*ldC] = beta*C[i+j*ldC] + _C[i+j*MR];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                C[i+j*ldC] = _C[i+j*MR];
            }
        }
     }
}

//
//  Micro Kernel for the Computation of _C <- _A*_B
//

/*
void
ULMBLAS(dgemm_micro_kernel)(const long    kc,
                            const double  alpha,
                            const double  *A,
                            const double  *B,
                            const double  beta,
                            double        *C,
                            const long    ldC,
                            const double  *,
                            const double  *)
{
    long    i, j, l;
    double register b;

    if (beta==0.0) {
        init_C();
    } else if (beta==1.0) {
        pack_C(MR, NR, C, ldC);
    } else {
        pack_bC(MR, NR, beta, C, ldC);
    }

//
//  Compute C <- C + alpha* _A*_B
//
    for (l=0; l<kc; ++l) {
        for (j=0; j<NR; ++j) {
            b = alpha*B[j+l*NR];
            for (i=0; i<MR; ++i) {
                _C[i+j*MR] += A[i+l*MR]*b;
            }
        }
    }
    unpack_C(MR, NR, C, ldC);
}
*/

//
//  4x4 Micro Kernel (assumes A is 4xk and  B is kx4)
//

/*
void
ULMBLAS(dgemm_micro_kernel)(const long    k,
                            const double  alpha,
                            const double  *A,
                            const double  *B,
                            const double  beta,
                            double        *C,
                            const long    ldC,
                            const double  *,
                            const double  *)
{
    double a0;
    double a1;
    double a2;
    double a3;

    unsigned long pA = (unsigned long) A;
    unsigned long pB = (unsigned long) B;

    if (pA%16 != 0) {
        fprintf(stderr, "A is not aligned\n");
        return;
    }

    if (pB%16 != 0) {
        fprintf(stderr, "B is not aligned\n");
        return;
    }

    double b0, b1, b2, b3;

    double ab00 = 0.0, ab01 = 0.0, ab02 = 0.0, ab03 = 0.0;
    double ab10 = 0.0, ab11 = 0.0, ab12 = 0.0, ab13 = 0.0;
    double ab20 = 0.0, ab21 = 0.0, ab22 = 0.0, ab23 = 0.0;
    double ab30 = 0.0, ab31 = 0.0, ab32 = 0.0, ab33 = 0.0;

    double *c00, *c01, *c02, *c03;
    double *c10, *c11, *c12, *c13;
    double *c20, *c21, *c22, *c23;
    double *c30, *c31, *c32, *c33;

    long i;

    c00 = &C[0+0*ldC];
    c10 = &C[1+0*ldC];
    c20 = &C[2+0*ldC];
    c30 = &C[3+0*ldC];

    c01 = &C[0+1*ldC];
    c11 = &C[1+1*ldC];
    c21 = &C[2+1*ldC];
    c31 = &C[3+1*ldC];

    c02 = &C[0+2*ldC];
    c12 = &C[1+2*ldC];
    c22 = &C[2+2*ldC];
    c32 = &C[3+2*ldC];

    c03 = &C[0+3*ldC];
    c13 = &C[1+3*ldC];
    c23 = &C[2+3*ldC];
    c33 = &C[3+3*ldC];

    for (i=0; i<k; ++i) {
        a0 = A[0];
        a1 = A[1];
        a2 = A[2];
        a3 = A[3];

        b0 = B[0];
        b1 = B[1];
        b2 = B[2];
        b3 = B[3];

        ab00 += a0*b0;
        ab10 += a1*b0;
        ab20 += a2*b0;
        ab30 += a3*b0;

        ab01 += a0*b1;
        ab11 += a1*b1;
        ab21 += a2*b1;
        ab31 += a3*b1;

        ab02 += a0*b2;
        ab12 += a1*b2;
        ab22 += a2*b2;
        ab32 += a3*b2;

        ab03 += a0*b3;
        ab13 += a1*b3;
        ab23 += a2*b3;
        ab33 += a3*b3;

        A += 4;
        B += 4;
    }
    if (beta == 0.0) {
        *c00 = 0.0;
        *c10 = 0.0;
        *c20 = 0.0;
        *c30 = 0.0;
        *c01 = 0.0;
        *c11 = 0.0;
        *c21 = 0.0;
        *c31 = 0.0;
        *c02 = 0.0;
        *c12 = 0.0;
        *c22 = 0.0;
        *c32 = 0.0;
        *c03 = 0.0;
        *c13 = 0.0;
        *c23 = 0.0;
        *c33 = 0.0;
    } else {
        *c00 *= beta;
        *c10 *= beta;
        *c20 *= beta;
        *c30 *= beta;
        *c01 *= beta;
        *c11 *= beta;
        *c21 *= beta;
        *c31 *= beta;
        *c02 *= beta;
        *c12 *= beta;
        *c22 *= beta;
        *c32 *= beta;
        *c03 *= beta;
        *c13 *= beta;
        *c23 *= beta;
        *c33 *= beta;
    }
    *c00 += alpha * ab00;
    *c10 += alpha * ab10;
    *c20 += alpha * ab20;
    *c30 += alpha * ab30;

    *c01 += alpha * ab01;
    *c11 += alpha * ab11;
    *c21 += alpha * ab21;
    *c31 += alpha * ab31;

    *c02 += alpha * ab02;
    *c12 += alpha * ab12;
    *c22 += alpha * ab22;
    *c32 += alpha * ab32;

    *c03 += alpha * ab03;
    *c13 += alpha * ab13;
    *c23 += alpha * ab23;
    *c33 += alpha * ab33;
}
*/

/*
void
ULMBLAS(dgemm_micro_kernel)(const long    k,
                            const double  _alpha,
                            const double  *A,
                            const double  *B,
                            const double  _beta,
                            double        *C,
                            const long    ldC,
                            const double  *nextA,
                            const double  *nextB)
{
    __m128d A0;
    __m128d A2;

    __m128d b;

    __m128d Ab00, Ab01, Ab02, Ab03;
    __m128d Ab20, Ab21, Ab22, Ab23;

    __m128d _C00, _C01, _C02, _C03;
    __m128d _C20, _C21, _C22, _C23;

    __m128d alpha, beta;

    long i;

    _mm_prefetch(nextA, 2);
    _mm_prefetch(nextB, 1);

    Ab00 = _mm_setzero_pd();
    Ab01 = _mm_setzero_pd();
    Ab02 = _mm_setzero_pd();
    Ab03 = _mm_setzero_pd();

    Ab20 = _mm_setzero_pd();
    Ab21 = _mm_setzero_pd();
    Ab22 = _mm_setzero_pd();
    Ab23 = _mm_setzero_pd();

    for (i=0; i<k; ++i, A+=4, B+=4) {
        A0 = _mm_load_pd(A);
        A2 = _mm_load_pd(A+2);

        b = _mm_load_pd1(B);
        Ab00 = _mm_add_pd(Ab00, _mm_mul_pd(A0, b));
        Ab20 = _mm_add_pd(Ab20, _mm_mul_pd(A2, b));

        b = _mm_load_pd1(B+1);
        Ab01 = _mm_add_pd(Ab01, _mm_mul_pd(A0, b));
        Ab21 = _mm_add_pd(Ab21, _mm_mul_pd(A2, b));

        b = _mm_load_pd1(B+2);
        Ab02 = _mm_add_pd(Ab02, _mm_mul_pd(A0, b));
        Ab22 = _mm_add_pd(Ab22, _mm_mul_pd(A2, b));

        b = _mm_load_pd1(B+3);
        Ab03 = _mm_add_pd(Ab03, _mm_mul_pd(A0, b));
        Ab23 = _mm_add_pd(Ab23, _mm_mul_pd(A2, b));
    }

    if (_beta == 0.0) {
        _C00 = _mm_setzero_pd();
        _C20 = _mm_setzero_pd();

        _C01 = _mm_setzero_pd();
        _C21 = _mm_setzero_pd();

        _C02 = _mm_setzero_pd();
        _C22 = _mm_setzero_pd();

        _C03 = _mm_setzero_pd();
        _C23 = _mm_setzero_pd();
    } else {
        _C00 = _mm_load_pd(&C[0+0*ldC]);
        _C20 = _mm_load_pd(&C[2+0*ldC]);

        _C01 = _mm_load_pd(&C[0+1*ldC]);
        _C21 = _mm_load_pd(&C[2+1*ldC]);

        _C02 = _mm_load_pd(&C[0+2*ldC]);
        _C22 = _mm_load_pd(&C[2+2*ldC]);

        _C03 = _mm_load_pd(&C[0+3*ldC]);
        _C23 = _mm_load_pd(&C[2+3*ldC]);

        beta = _mm_load_pd1(&_beta);

        _C00 = _mm_mul_pd(beta, _C00);
        _C20 = _mm_mul_pd(beta, _C20);

        _C01 = _mm_mul_pd(beta, _C01);
        _C21 = _mm_mul_pd(beta, _C21);

        _C02 = _mm_mul_pd(beta, _C02);
        _C22 = _mm_mul_pd(beta, _C22);

        _C03 = _mm_mul_pd(beta, _C03);
        _C23 = _mm_mul_pd(beta, _C23);
    }
    alpha = _mm_load_pd1(&_alpha);

    _C00 = _mm_add_pd(_C00, _mm_mul_pd(alpha, Ab00));
    _C20 = _mm_add_pd(_C20, _mm_mul_pd(alpha, Ab20));

    _C01 = _mm_add_pd(_C01, _mm_mul_pd(alpha, Ab01));
    _C21 = _mm_add_pd(_C21, _mm_mul_pd(alpha, Ab21));

    _C02 = _mm_add_pd(_C02, _mm_mul_pd(alpha, Ab02));
    _C22 = _mm_add_pd(_C22, _mm_mul_pd(alpha, Ab22));

    _C03 = _mm_add_pd(_C03, _mm_mul_pd(alpha, Ab03));
    _C23 = _mm_add_pd(_C23, _mm_mul_pd(alpha, Ab23));

    _mm_store_pd(&C[0+0*ldC], _C00);
    _mm_store_pd(&C[2+0*ldC], _C20);

    _mm_store_pd(&C[0+1*ldC], _C01);
    _mm_store_pd(&C[2+1*ldC], _C21);

    _mm_store_pd(&C[0+2*ldC], _C02);
    _mm_store_pd(&C[2+2*ldC], _C22);

    _mm_store_pd(&C[0+3*ldC], _C03);
    _mm_store_pd(&C[2+3*ldC], _C23);
}
*/

//extern long count_dgemm_micro_kernel;

/*
void
ULMBLAS(dgemm_micro_kernel)(const long    k,
                            const double  * restrict alpha,
                            const double  * restrict a,
                            const double  * restrict b,
                            const double  * restrict beta,
                            double        * restrict c,
                            const long    ldC,
                            const void    *a_next,
                            const void    *b_next)
*/

typedef struct
{
    const void*  a_next;
    const void*  b_next;

    // The panel strides of A and B.
    long  ps_a;
    long  ps_b;

} auxinfo_t;

void
ULMBLAS(dgemm_micro_kernel)(long                    k,
                            const double* restrict  alpha,
                            const double* restrict  a,
                            const double* restrict  b,
                            const double* restrict  beta,
                            double* restrict        c,
                            long                    rs_c,
                            long                    cs_c,
                            auxinfo_t*              data)
{
    const void *a_next = data->a_next;
    const void *b_next = data->b_next;

    const long k_iter = k / 4;
    const long k_left = k % 4;

    /*
    const long rs_c = 1;
    const long cs_c = ldC;

    count_dgemm_micro_kernel += k;

    fprintf(stderr, "a_next-a = %ld, b_next -b =%ld\n",
            (unsigned long)(nextA -a),
            (unsigned long)(nextB -b));
    */

    __asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"movq          %2, %%rax         \n\t" // load address of a.
		"movq          %3, %%rbx         \n\t" // load address of b.
		"movq          %9, %%r9          \n\t" // load address of b_next.
		"movq         %10, %%r11         \n\t" // load address of a_next.
		"                                \n\t"
		"subq    $-8 * 16, %%rax         \n\t" // increment pointers to allow byte
		"subq    $-8 * 16, %%rbx         \n\t" // offsets in the unrolled iterations.
		"                                \n\t"
		"movaps  -8 * 16(%%rax), %%xmm0  \n\t" // initialize loop by pre-loading elements
		"movaps  -7 * 16(%%rax), %%xmm1  \n\t" // of a and b.
		"movaps  -8 * 16(%%rbx), %%xmm2  \n\t"
		"                                \n\t"
		"movq          %6, %%rcx         \n\t" // load address of c
		"movq          %8, %%rdi         \n\t" // load cs_c
		"leaq        (,%%rdi,8), %%rdi   \n\t" // cs_c *= sizeof(double)
		"leaq   (%%rcx,%%rdi,2), %%r10   \n\t" // load address of c + 2*cs_c;
		"                                \n\t"
		"prefetcht2   0 * 8(%%r9)        \n\t" // prefetch b_next
		"                                \n\t"
		"xorpd     %%xmm3,  %%xmm3       \n\t"
		"xorpd     %%xmm4,  %%xmm4       \n\t"
		"xorpd     %%xmm5,  %%xmm5       \n\t"
		"xorpd     %%xmm6,  %%xmm6       \n\t"
		"                                \n\t"
		"prefetcht2   3 * 8(%%rcx)       \n\t" // prefetch c + 0*cs_c
		"xorpd     %%xmm8,  %%xmm8       \n\t"
		"movaps    %%xmm8,  %%xmm9       \n\t"
		"prefetcht2   3 * 8(%%rcx,%%rdi) \n\t" // prefetch c + 1*cs_c
		"movaps    %%xmm8, %%xmm10       \n\t"
		"movaps    %%xmm8, %%xmm11       \n\t"
		"prefetcht2   3 * 8(%%r10)       \n\t" // prefetch c + 2*cs_c
		"movaps    %%xmm8, %%xmm12       \n\t"
		"movaps    %%xmm8, %%xmm13       \n\t"
		"prefetcht2   3 * 8(%%r10,%%rdi) \n\t" // prefetch c + 3*cs_c
		"movaps    %%xmm8, %%xmm14       \n\t"
		"movaps    %%xmm8, %%xmm15       \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movq      %0, %%rsi             \n\t" // i = k_iter;
		"testq  %%rsi, %%rsi             \n\t" // check i via logical AND.
		"je     .DCONSIDKLEFT            \n\t" // if i == 0, jump to code that
		"                                \n\t" // contains the k_left loop.
		"                                \n\t"
		"                                \n\t"
		".DLOOPKITER:                    \n\t" // MAIN LOOP
		"                                \n\t"
		"prefetcht0  (4*35+1) * 8(%%rax) \n\t"
		//"prefetcht0  (8*97+4) * 8(%%rax) \n\t"
		"                                \n\t"
		//"prefetcht0  67*4 * 8(%%r11)       \n\t" // prefetch a_next[0]
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 0
		"movaps  -7 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -6 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -6 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -5 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 1
		"movaps  -5 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -4 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -4 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -3 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"prefetcht0  (4*37+1) * 8(%%rax) \n\t"
		//"prefetcht0  (8*97+12)* 8(%%rax) \n\t"
		"                                \n\t"
		//"prefetcht0  69*4 * 8(%%r11)       \n\t" // prefetch a_next[8]
		//"subq  $-4 * 4 * 8, %%r11        \n\t" // a_next += 4*4 (unroll x mr)
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 2
		"movaps  -3 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -2 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -2 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -1 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 3
		"movaps  -1 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"subq  $-4 * 4 * 8, %%rax        \n\t" // a += 4*4 (unroll x mr)
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"subq  $-4 * 4 * 8, %%r9         \n\t" // b_next += 4*4 (unroll x nr)
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps   0 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"subq  $-4 * 4 * 8, %%rbx        \n\t" // b += 4*4 (unroll x nr)
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -8 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -7 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"prefetcht2        0 * 8(%%r9)   \n\t" // prefetch b_next[0]
		"prefetcht2        8 * 8(%%r9)   \n\t" // prefetch b_next[8]
		"                                \n\t"
		"decq   %%rsi                    \n\t" // i -= 1;
		"jne    .DLOOPKITER              \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		//"prefetcht2       -8 * 8(%%r9)   \n\t" // prefetch b_next[-8]
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDKLEFT:                  \n\t"
		"                                \n\t"
		"movq      %1, %%rsi             \n\t" // i = k_left;
		"testq  %%rsi, %%rsi             \n\t" // check i via logical AND.
		"je     .DPOSTACCUM              \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"                                \n\t"
		".DLOOPKLEFT:                    \n\t" // EDGE LOOP
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 0
		"movaps  -7 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -6 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -6 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -5 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"subq  $-4 * 1 * 8, %%rax        \n\t" // a += 4 (1 x mr)
		"subq  $-4 * 1 * 8, %%rbx        \n\t" // b += 4 (1 x nr)
		"                                \n\t"
		"                                \n\t"
		"decq   %%rsi                    \n\t" // i -= 1;
		"jne    .DLOOPKLEFT              \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM:                    \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"                                \n\t"
		"                                \n\t"
		"movq    %4, %%rax               \n\t" // load address of alpha
		"movq    %5, %%rbx               \n\t" // load address of beta 
		"movddup (%%rax), %%xmm6         \n\t" // load alpha and duplicate
		"movddup (%%rbx), %%xmm7         \n\t" // load beta and duplicate
		"                                \n\t"
		"                                \n\t"
		"movq    %7, %%rsi               \n\t" // load rs_c
		"movq    %%rsi, %%r8             \n\t" // make a copy of rs_c
		"                                \n\t"
		"leaq    (,%%rsi,8), %%rsi       \n\t" // rsi = rs_c * sizeof(double)
		"                                \n\t"
		"leaq   (%%rcx,%%rsi,2), %%rdx   \n\t" // load address of c + 2*rs_c;
		"                                \n\t"
		"                                \n\t" // xmm8:   xmm9:   xmm10:  xmm11:
		"                                \n\t" // ( ab01  ( ab00  ( ab03  ( ab02
		"                                \n\t" //   ab10 )  ab11 )  ab12 )  ab13 )
		"                                \n\t" //
		"                                \n\t" // xmm12:  xmm13:  xmm14:  xmm15:
		"                                \n\t" // ( ab21  ( ab20  ( ab23  ( ab22
		"                                \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
		"movaps   %%xmm8,  %%xmm0        \n\t"
		"movsd    %%xmm9,  %%xmm8        \n\t"
		"movsd    %%xmm0,  %%xmm9        \n\t"
		"                                \n\t"
		"movaps  %%xmm10,  %%xmm0        \n\t"
		"movsd   %%xmm11, %%xmm10        \n\t"
		"movsd    %%xmm0, %%xmm11        \n\t"
		"                                \n\t"
		"movaps  %%xmm12,  %%xmm0        \n\t"
		"movsd   %%xmm13, %%xmm12        \n\t"
		"movsd    %%xmm0, %%xmm13        \n\t"
		"                                \n\t"
		"movaps  %%xmm14,  %%xmm0        \n\t"
		"movsd   %%xmm15, %%xmm14        \n\t"
		"movsd    %%xmm0, %%xmm15        \n\t"
		"                                \n\t" // xmm8:   xmm9:   xmm10:  xmm11:
		"                                \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
		"                                \n\t" //   ab10 )  ab11 )  ab12 )  ab13 )
		"                                \n\t" //
		"                                \n\t" // xmm12:  xmm13:  xmm14:  xmm15:
		"                                \n\t" // ( ab20  ( ab21  ( ab22  ( ab23
		"                                \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t" // determine if
		"                                \n\t" //   c % 16 == 0, AND
		"                                \n\t" //   rs_c == 1
		"                                \n\t" // ie: aligned and column-stored
		"                                \n\t"
		"cmpq       $1, %%r8             \n\t" // set ZF if rs_c == 1.
		"sete           %%bl             \n\t" // bl = ( ZF == 1 ? 1 : 0 );
		"testq     $15, %%rcx            \n\t" // set ZF if c & 16 is zero.
		"setz           %%bh             \n\t" // bh = ( ZF == 1 ? 1 : 0 );
		"                                \n\t" // and(bl,bh) will reveal result
		"                                \n\t"
		"                                \n\t" // now avoid loading C if beta == 0
		"                                \n\t"
		"xorpd     %%xmm0,  %%xmm0       \n\t" // set xmm0 to zero.
		"ucomisd   %%xmm0,  %%xmm7       \n\t" // check if beta == 0.
		"je      .DBETAZERO              \n\t" // if ZF = 1, jump to beta == 0 case
		"                                \n\t"
		"                                \n\t"
		"                                \n\t" // check if aligned/column-stored
		"andb     %%bl, %%bh             \n\t" // set ZF if bl & bh == 1.
		"jne     .DCOLSTORED             \n\t" // jump to column storage case
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DGENSTORED:                    \n\t"
		"                                \n\t"
		"movlpd  (%%rcx),       %%xmm0   \n\t" // load c00 and c10,
		"movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
		"mulpd   %%xmm6,  %%xmm8         \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd   %%xmm8,  %%xmm0         \n\t" // add the gemm result,
		"movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movlpd  (%%rdx),       %%xmm1   \n\t" // load c20 and c30,
		"movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
		"mulpd   %%xmm6,  %%xmm12        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm12,  %%xmm1         \n\t" // add the gemm result,
		"movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movlpd  (%%rcx),       %%xmm0   \n\t" // load c01 and c11,
		"movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
		"mulpd   %%xmm6,  %%xmm9         \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd   %%xmm9,  %%xmm0         \n\t" // add the gemm result,
		"movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movlpd  (%%rdx),       %%xmm1   \n\t" // load c21 and c31,
		"movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
		"mulpd   %%xmm6,  %%xmm13        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm13,  %%xmm1         \n\t" // add the gemm result,
		"movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movlpd  (%%rcx),       %%xmm0   \n\t" // load c02 and c12,
		"movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
		"mulpd   %%xmm6,  %%xmm10        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd  %%xmm10,  %%xmm0         \n\t" // add the gemm result,
		"movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movlpd  (%%rdx),       %%xmm1   \n\t" // load c22 and c32,
		"movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
		"mulpd   %%xmm6,  %%xmm14        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm14,  %%xmm1         \n\t" // add the gemm result,
		"movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movlpd  (%%rcx),       %%xmm0   \n\t" // load c03 and c13,
		"movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
		"mulpd   %%xmm6,  %%xmm11        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd  %%xmm11,  %%xmm0         \n\t" // add the gemm result,
		"movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
		"                                \n\t"
		"                                \n\t"
		"movlpd  (%%rdx),       %%xmm1   \n\t" // load c23 and c33,
		"movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
		"mulpd   %%xmm6,  %%xmm15        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm15,  %%xmm1         \n\t" // add the gemm result,
		"movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
		"                                \n\t"
		"jmp    .DDONE                   \n\t" // jump to end.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCOLSTORED:                    \n\t"
		"                                \n\t"
		"movaps  (%%rcx),       %%xmm0   \n\t" // load c00 and c10,
		"mulpd   %%xmm6,  %%xmm8         \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd   %%xmm8,  %%xmm0         \n\t" // add the gemm result,
		"movaps  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movaps  (%%rdx),       %%xmm1   \n\t" // load c20 and c30,
		"mulpd   %%xmm6,  %%xmm12        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm12,  %%xmm1         \n\t" // add the gemm result,
		"movaps  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movaps  (%%rcx),       %%xmm0   \n\t" // load c01 and c11,
		"mulpd   %%xmm6,  %%xmm9         \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd   %%xmm9,  %%xmm0         \n\t" // add the gemm result,
		"movaps  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movaps  (%%rdx),       %%xmm1   \n\t" // load c21 and c31,
		"mulpd   %%xmm6,  %%xmm13        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm13,  %%xmm1         \n\t" // add the gemm result,
		"movaps  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movaps  (%%rcx),       %%xmm0   \n\t" // load c02 and c12,
		"mulpd   %%xmm6,  %%xmm10        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd  %%xmm10,  %%xmm0         \n\t" // add the gemm result,
		"movaps  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movaps  (%%rdx),       %%xmm1   \n\t" // load c22 and c32,
		"mulpd   %%xmm6,  %%xmm14        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm14,  %%xmm1         \n\t" // add the gemm result,
		"movaps  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movaps  (%%rcx),       %%xmm0   \n\t" // load c03 and c13,
		"mulpd   %%xmm6,  %%xmm11        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd  %%xmm11,  %%xmm0         \n\t" // add the gemm result,
		"movaps  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"                                \n\t"
		"                                \n\t"
		"movaps  (%%rdx),       %%xmm1   \n\t" // load c23 and c33,
		"mulpd   %%xmm6,  %%xmm15        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm15,  %%xmm1         \n\t" // add the gemm result,
		"movaps  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"                                \n\t"
		"jmp    .DDONE                   \n\t" // jump to end.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DBETAZERO:                     \n\t"
		"                                \n\t" // check if aligned/column-stored
		"andb     %%bl, %%bh             \n\t" // set ZF if bl & bh == 1.
		"jne     .DCOLSTORBZ             \n\t" // jump to column storage case
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DGENSTORBZ:                    \n\t"
		"                                \n\t" // skip loading c00 and c10,
		"mulpd   %%xmm6,  %%xmm8         \n\t" // scale by alpha,
		"movlpd  %%xmm8,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm8,  (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t" // skip loading c20 and c30,
		"mulpd   %%xmm6,  %%xmm12        \n\t" // scale by alpha,
		"movlpd  %%xmm12, (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm12, (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t" // skip loading c01 and c11,
		"mulpd   %%xmm6,  %%xmm9         \n\t" // scale by alpha,
		"movlpd  %%xmm9,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm9,  (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t" // skip loading c21 and c31,
		"mulpd   %%xmm6,  %%xmm13        \n\t" // scale by alpha,
		"movlpd  %%xmm13, (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm13, (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t" // skip loading c02 and c12,
		"mulpd   %%xmm6,  %%xmm10        \n\t" // scale by alpha,
		"movlpd  %%xmm10, (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm10, (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t" // skip loading c22 and c32,
		"mulpd   %%xmm6,  %%xmm14        \n\t" // scale by alpha,
		"movlpd  %%xmm14, (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm14, (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t" // skip loading c03 and c13,
		"mulpd   %%xmm6,  %%xmm11        \n\t" // scale by alpha,
		"movlpd  %%xmm11, (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm11, (%%rcx,%%rsi)  \n\t"
		"                                \n\t"
		"                                \n\t" // skip loading c23 and c33,
		"mulpd   %%xmm6,  %%xmm15        \n\t" // scale by alpha,
		"movlpd  %%xmm15, (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm15, (%%rdx,%%rsi)  \n\t"
		"                                \n\t"
		"jmp    .DDONE                   \n\t" // jump to end.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCOLSTORBZ:                    \n\t"
		"                                \n\t"
		"                                \n\t" // skip loading c00 and c10,
		"mulpd   %%xmm6,  %%xmm8         \n\t" // scale by alpha,
		"movaps  %%xmm8,  (%%rcx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t" // skip loading c20 and c30,
		"mulpd   %%xmm6,  %%xmm12        \n\t" // scale by alpha,
		"movaps  %%xmm12, (%%rdx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t" // skip loading c01 and c11,
		"mulpd   %%xmm6,  %%xmm9         \n\t" // scale by alpha,
		"movaps  %%xmm9,  (%%rcx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t" // skip loading c21 and c31,
		"mulpd   %%xmm6,  %%xmm13        \n\t" // scale by alpha,
		"movaps  %%xmm13, (%%rdx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t" // skip loading c02 and c12,
		"mulpd   %%xmm6,  %%xmm10        \n\t" // scale by alpha,
		"movaps  %%xmm10, (%%rcx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t" // skip loading c22 and c32,
		"mulpd   %%xmm6,  %%xmm14        \n\t" // scale by alpha,
		"movaps  %%xmm14, (%%rdx)        \n\t" // and store back to memory.
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t" // skip loading c03 and c13,
		"mulpd   %%xmm6,  %%xmm11        \n\t" // scale by alpha,
		"movaps  %%xmm11, (%%rcx)        \n\t" // and store back to memory.
		"                                \n\t"
		"                                \n\t" // skip loading c23 and c33,
		"mulpd   %%xmm6,  %%xmm15        \n\t" // scale by alpha,
		"movaps  %%xmm15, (%%rdx)        \n\t" // and store back to memory.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DDONE:                         \n\t"
		"                                \n\t"

		: // output operands (none)
		: // input operands
		  "m" (k_iter), // 0
		  "m" (k_left), // 1
		  "m" (a),      // 2
		  "m" (b),      // 3
		  "m" (alpha),  // 4
		  "m" (beta),   // 5
		  "m" (c),      // 6
		  "m" (rs_c),   // 7
		  "m" (cs_c),   // 8
		  "m" (b_next), // 9
		  "m" (a_next)  // 10
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	);
}


//
//  Macro Kernel for the Computation of C <- C + _A*_B
//

void
ULMBLAS(dgemm_macro_kernel)(const long     mc,
                            const long     nc,
                            const long     kc,
                            const double   alpha,
                            const double   beta,
                            double         * restrict C,
                            const long     ldC)
{
    const long M = (mc+MR-1) / MR;
    const long N = (nc+NR-1) / NR;

    const long _MR = mc % MR;
    const long _NR = nc % NR;

    const double * restrict a, * restrict aNext;
    const double * restrict b, * restrict bNext;
    double       * restrict cj, * restrict cij;

    //const long alignedC = ((unsigned long)C % 16 == 0) && (ldC % 4 == 0);

    long  I, J, mr, nr;

    const double zero = 0.0;

    const double * restrict palpha = &alpha;
    const double * restrict pbeta = &beta;

    const long incA = MR*kc;
    const long incB = NR*kc;
    const long incRowC = MR;
    const long incColC = NR*ldC;

    auxinfo_t  data;

    data.ps_a = incA;
    data.ps_b = incB;

    b = _B;
    cj = C;

    for (J=0; J<N; ++J) {
        nr = (J!=N-1 || _NR==0) ? NR : _NR;

        a     = _A;
        cij   = cj;
        bNext = b;

        for (I=0; I<M; ++I) {
            mr = (I!=M-1 || _MR==0) ? MR : _MR;

            aNext = a + incA;
            if (I==M-1) {
                aNext = _A;
                bNext = b + incB;
                if (J==N-1) {
                    bNext = _B;
                }
            }

            data.a_next = aNext;
            data.b_next = bNext;

            if (mr==MR && nr==NR) {
                ULMBLAS(dgemm_micro_kernel)(kc,
                                            palpha,
                                            a,
                                            b,
                                            pbeta,
                                            cij,
                                            1,
                                            ldC,
                                            &data);
                /*
                ULMBLAS(dgemm_micro_kernel)(kc,
                                            palpha,
                                            a, b,
                                            pbeta,
                                            cij, ldC,
                                            aNext, bNext);
                */
            } else {
                ULMBLAS(dgemm_micro_kernel)(kc,
                                            palpha,
                                            a,
                                            b,
                                            &zero,
                                            cij,
                                            1,
                                            ldC,
                                            &data);
                 /*
                ULMBLAS(dgemm_micro_kernel)(kc,
                                            palpha,
                                            a, b,
                                            &zero,
                                            _C, MR,
                                            aNext, bNext);
                ULMBLAS(unpack_C)(mr, nr, beta, cij, ldC);
                */
            }
            a+=incA;
            cij+=incRowC;
        }
        b+=incB;
        cj+=incColC;
    }
}

//
//  Computation of C <- beta*C + alpha*A*B
//
void
ULMBLAS(dgemm_nn)(const long        m,
                  const long        n,
                  const long        k,
                  const double      alpha,
                  const double      * restrict A,
                  const long        ldA,
                  const double      * restrict B,
                  const long        ldB,
                  const double      beta,
                  double            * restrict C,
                  const long        ldC)
{
//
//  Number of panels
//
    const long N = (n+NC-1) / NC;
    const long K = (k+KC-1) / KC;
    const long M = (m+MC-1) / MC;

//
//  Width/height of panels at the bottom or right side
//
    const long _NC = n % NC;
    const long _KC = k % KC;
    const long _MC = m % MC;

//
//  For holding the actual panel width/height
//
    long mc, nc, kc;

//
//  Upper case letters are used for indexing matrix panels.  Lower case letters
//  are used for indexing matrix elements.
//
    long J, L, I;
    long j, l, i;

//
//  Start the operation on the macro level
//
    for (J=0, j=0; J<N; ++J, j+=NC) {
        nc = (J!=N-1 || _NC==0) ? NC : _NC;

        for (L=0, l=0; L<K; ++L, l+=KC) {
            kc = (L!=K-1 || _KC==0) ? KC : _KC;

//
//          Pack matrix block alpha*B(l:l+kc-1,j:j+nc-1) into buffer _B
//
            ULMBLAS(pack_B)(kc, nc, &B[l+j*ldB], ldB);

            for (I=0, i=0; I<M; ++I, i+=MC) {
                mc = (I!=M-1 || _MC==0) ? MC : _MC;

//
//              Pack block A(i:i+mc-1,l:l+kc-1) into buffer _A
//
                ULMBLAS(pack_A)(mc, kc, &A[i+l*ldA], ldA);

//
//              C(i:i+mc,j:j+nc-1) <- C(i:i+mc,j:j+nc-1) + _A*_B
//
                ULMBLAS(dgemm_macro_kernel)(mc, nc, kc,
                                            alpha,
                                            beta,
                                            &C[i+j*ldC], ldC);
            }
        }
    }
}
