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

//
//  Packing, scaling  and padding panels from A into macro kernel buffer _A
//

void
pack_A(const int      m,
       const int      n,
       const double   *A,
       const int      ldA)
{
//
//  m = M*MR + _MR with (0<=_MR<MR)
//
    const int M   = m / MR;
    const int _MR = m % MR;

    //double *p = _A;
    int    i, j, I;

    for (j=0; j<n; ++j) {
        double *p = &_A[j*MR];
        const double *q = &A[j*ldA];
        for (I=0; I<M; ++I, p+=n*MR, q+=MR) {
            for (i=0; i<MR; ++i) {
                p[i] = q[i];
            }
        }
    }
    if (_MR>0) {
        double *p = _A;
        for (j=0; j<n; ++j) {
            for (i=0; i<_MR; ++i) {
                p[i+(j+n*M)*MR] = A[i+M*MR+j*ldA];
            }
            for (i=_MR; i<MR; ++i) {
                p[i+(j+n*M)*MR] = 0.0;
            }
        }
    }
}

//
//  Packing and padding panels from B into macro kernel buffer _B
//

void
pack_B(const int      m,
       const int      n,
       const double   *B,
       const int      ldB)
{
//
//  n = N*NR + _NR with (0<=_NR<NR)
//
    const int N   = n / NR;
    const int _NR = n % NR;

    int    i, j, J;

    double       *p = _B;
    const double *q = B;

    for (J=0; J<N; ++J, q+=NR*ldB) {
        for (i=0; i<m; ++i, p+=NR) {
            const double *_q = &q[i];
            for (j=0; j<NR; ++j) {
                p[j] = _q[j*ldB];
            }
        }
    }
    if (_NR>0) {
        double *p = _B;
        for (j=0; j<_NR; ++j) {
            for (i=0; i<m; ++i) {
                p[NR*(m*N+i)+j] = B[i+(N*NR+j)*ldB];
            }
        }
        for (j=_NR; j<NR; ++j) {
            for (i=0; i<m; ++i) {
                p[NR*(m*N+i)+j] = 0.0;
            }
        }
    }
}
//
//  Pack C into buffer _C
//

void
init_C()
{
    int i, j;

    for (j=0; j<NR; ++j) {
        for (i=0; i<MR; ++i) {
            _C[i+MR*j] = 0.0;
        }
    }
}

void
pack_C(const int     m,
       const int     n,
       double        *C,
       const int     ldC)
{
    int i, j;

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            _C[i+j*MR] = C[i+j*ldC];
        }
    }
}

void
pack_bC(const int     m,
       const int     n,
       const double  beta,
       double        *C,
       const int     ldC)
{
    int i, j;

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            _C[i+j*MR] = beta*C[i+j*ldC];
        }
    }
}


//
//  Unpack C from buffer _C
//

void
unpack_C(const int     m,
         const int     n,
         const double  beta,
         double        *C,
         const int     ldC)
{
    int i, j;

    if (beta!=0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                C[i+j*ldC] = beta*C[i+j*ldC] + _C[i+j*MR];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                C[i+j*ldC] = 2* _C[i+j*MR];
            }
        }
     }
}

//
//  Scale matrix X
//

void
dscale(const int    m,
      const int     n,
      const double  alpha,
      double        *X,
      const int     ldX)
{
    int i,j;

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            X[i+j*ldX] *= alpha;
        }
    }
}

//
//  Micro Kernel for the Computation of _C <- _A*_B
//

/*
void
dgemm_micro_kernel(const int     kc,
                   const double  alpha,
                   const double  *A,
                   const double  *B,
                   const double  beta,
                   double        *C,
                   const int     ldC,
                   const double  *,
                   const double  *)
{
    int    i, j, l;
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
dgemm_micro_kernel(const int     k,
                   const double  alpha,
                   const double  *A,
                   const double  *B,
                   const double  beta,
                   double        *C,
                   const int     ldC,
                   const double  *,
                   const double  *)
{
    double a0;
    double a1;
    double a2;
    double a3;

    unsigned int pA = (unsigned long) A;
    unsigned int pB = (unsigned long) B;

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

    int i;

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
dgemm_micro_kernel(const int     k,
                   const double  _alpha,
                   const double  *A,
                   const double  *B,
                   const double  _beta,
                   double        *C,
                   const int     ldC,
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

    int i;

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

void
dgemm_micro_kernel(const long    k,
                   const double  _alpha,
                   const double  *a,
                   const double  *b,
                   const double  _beta,
                   double        *c,
                   const long    ldC,
                   const double  *nextA,
                   const double  *nextB);


//
//  Macro Kernel for the Computation of C <- C + _A*_B
//

void
dgemm_macro_kernel(const int      mc,
                   const int      nc,
                   const int      kc,
                   const double   alpha,
                   const double   beta,
                   double         *C,
                   const int      ldC)
{
    const int M = mc / MR;
    const int N = nc / NR;

    const int _MR = mc % MR;
    const int _NR = nc % NR;

    const double *a, *aNext;
    const double *b, *bNext;

    const int alignedC = ((unsigned long)C % 16 == 0) && (ldC % 4 == 0);

    int i, j, I, J, mr, nr;

    for (J=0, j=0, b=_B; J<=N; ++J, j+=NR, b+=NR*kc) {
        nr = (J<N) ? NR : _NR;
        for (I=0, i=0, a=_A; I<=M; ++I, i+=MR, a+=MR*kc) {
            mr = (I<M) ? MR : _MR;

            aNext = a;
            bNext = b;
            if (I<M) {
                aNext = a + MR*kc;
            } else {
                if (J<N) {
                    bNext = b + NR*kc;
                }
            }

            if (alignedC && J<N && I<M) {
                dgemm_micro_kernel(kc,
                                   alpha,
                                   a, b,
                                   beta,
                                   &C[i+j*ldC], ldC,
                                   aNext, bNext);
            } else {
                //pack_C(mr, nr, &C[i+j*ldC], ldC);
                dgemm_micro_kernel(kc,
                                   alpha,
                                   a, b,
                                   0.0,
                                   _C, MR,
                                   aNext, bNext);
                unpack_C(mr, nr, beta, &C[i+j*ldC], ldC);
            }
        }
    }
}

//
//  Computation of C <- beta*C + alpha*A*B
//

void
ULMBLAS(dgemm_nn)(const int         m,
                  const int         n,
                  const int         k,
                  const double      alpha,
                  const double      *A,
                  const int         ldA,
                  const double      *B,
                  const int         ldB,
                  const double      beta,
                  double            *C,
                  const int         ldC)
{
//
//  Number of panels
//
    const int N = n / NC;
    const int K = k / KC;
    const int M = m / MC;

//
//  Width/height of panels at the bottom or right side
//
    const int _NC = n % NC;
    const int _KC = k % KC;
    const int _MC = m % MC;

//
//  For holding the actual panel width/height
//
    int mc, nc, kc;

//
//  Upper case letters are used for indexing matrix panels.  Lower case letters
//  are used for indexing matrix elements.
//
    int J, L, I;
    int j, l, i;

//
//  Start the operation on the macro level
//
    for (J=0, j=0; J<=N; ++J, j+=NC) {
        nc = (J<N) ? NC : _NC;

        for (L=0, l=0; L<=K; ++L, l+=KC) {
            kc = (L<K) ? KC : _KC;
//
//          Pack matrix block alpha*B(l:l+kc-1,j:j+nc-1) into buffer _B
//
            //fprintf(stderr, "pack B(%d:%d,%d:%d)\n", l, l+kc-1, j, j+nc-1);
            pack_B(kc, nc, &B[l+j*ldB], ldB);

            for (I=0, i=0; I<=M; ++I, i+=MC) {
                mc = (I<M) ? MC : _MC;
//
//              Pack block A(i:i+mc-1,l:l+kc-1) into buffer _A
//
                //fprintf(stderr, "pack A(%d:%d,%d:%d)\n", i, i+mc-1, l, l+kc-1);
                pack_A(mc, kc, &A[i+l*ldA], ldA);

//
//              Before the micro kernel does any computation update
//              C(i:i+mc,j:j+nc-1) <- beta*C(i:i+mc,j:j+nc-1)
//
//              if (L==0 && beta!=1.0) {
//                  dscale(mc, nc, beta, &C[i+j*ldC], ldC);
//              }

//
//              C(i:i+mc,j:j+nc-1) <- C(i:i+mc,j:j+nc-1) + _A*_B
//
                dgemm_macro_kernel(mc, nc, kc, alpha, beta, &C[i+j*ldC], ldC);
            }
        }
    }
}
