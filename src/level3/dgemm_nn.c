#include <ulmblas.h>
#include <stdio.h>

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
static double _C[MC*NC] __attribute__ ((aligned (16)));

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

    double *p = _A;
    int    i, j, I;

    for (j=0; j<n; ++j) {
        for (I=0; I<M; ++I) {
            for (i=0; i<MR; ++i) {
                p[i+(j+n*I)*MR] = A[i+I*MR+j*ldA];
            }
        }
    }
    if (_MR>0) {
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
       const double   alpha,
       const double   *B,
       const int      ldB)
{
//
//  n = N*NR + _NR with (0<=_NR<NR)
//
    const int N   = n / NR;
    const int _NR = n % NR;

    int    i, j, J;

    double *p = _B;

    if (alpha!=1.0) {
        for (J=0; J<N; ++J) {
            for (j=0; j<NR; ++j) {
                for (i=0; i<m; ++i) {
                    p[NR*(m*J+i)+j] = alpha*B[i+(J*NR+j)*ldB];
                }
            }
        }
        if (_NR>0) {
            for (j=0; j<_NR; ++j) {
                for (i=0; i<m; ++i) {
                    p[NR*(m*N+i)+j] = alpha*B[i+(N*NR+j)*ldB];
                }
            }
            for (j=_NR; j<NR; ++j) {
                for (i=0; i<m; ++i) {
                    p[NR*(m*N+i)+j] = 0.0;
                }
            }
        }
    } else {
        for (J=0; J<N; ++J) {
            for (j=0; j<NR; ++j) {
                for (i=0; i<m; ++i) {
                    p[NR*(m*J+i)+j] = B[i+(J*NR+j)*ldB];
                }
            }
        }
        if (_NR>0) {
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
}

//
//  Unpack and update C from buffer _C
//

void
unpack_C(const int     m,
         const int     n,
         double        *C,
         const int     ldC)
{
    int i, j;

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            C[i+j*ldC] += _C[i+j*MR];
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

void
dgemm_micro_kernel(const int     kc,
                   const double  *A,
                   const double  *B,
                   double        *C)
{
    int    i, j, l;
    double b;

//
//  Initialize _C with zeros
//
    for (j=0; j<NR; ++j) {
        for (i=0; i<MR; ++i) {
            C[i+j*MR] = 0.0;
        }
    }

//
//  Compute _C = _A*_B
//
    for (l=0; l<kc; ++l) {
        for (j=0; j<NR; ++j) {
            b = B[j+l*NR];
            for (i=0; i<MR; ++i) {
                C[i+j*MR] += A[i+l*MR]*b;
            }
        }
    }
}

//
//  Macro Kernel for the Computation of C <- C + _A*_B
//

void
dgemm_macro_kernel(const int      mc,
                   const int      nc,
                   const int      kc,
                   double         *C,
                   const int      ldC)
{
    const int M = mc / MR;
    const int N = nc / NR;

    const int _MR = mc % MR;
    const int _NR = nc % NR;

    const double *a;
    const double *b;

    int i, j, I, J, mr, nr;

    for (I=0, i=0, a=_A; I<=M; ++I, i+=MR, a+=MR*kc) {
        mr = (I<M) ? MR : _MR;
        for (J=0, j=0, b=_B; J<=N; ++J, j+=NR, b+=NR*kc) {
            nr = (J<N) ? NR : _NR;
            dgemm_micro_kernel(kc, a, b, _C);
            unpack_C(mr, nr, &C[i+j*ldC], ldC);
        }
    }
}

//
//  Computation of C <- beta*C + alpha*A*B
//

void
dgemm_nn(const int         m,
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
            pack_B(kc, nc, alpha, &B[l+j*ldB], ldB);

            for (I=0, i=0; I<=M; ++I, i+=MC) {
                mc = (I<M) ? MC : _MC;
//
//              Pack block A(i:i+mc-1,l:l+kc-1) into buffer _A
//
                pack_A(mc, kc, &A[i+l*ldA], ldA);

//
//              Before the micro kernel does any computation initialize
//              C(i:i+mc,j:j+nc-1) <- beta*C(i:i+mc,j:j+nc-1)
//
                if (L==0 && beta!=1.0) {
                    dscale(mc, nc, beta, &C[i+j*ldC], ldC);
                }

//
//              C(i:i+mc,j:j+nc-1) <- C(i:i+mc,j:j+nc-1) + _A*_B
//
                dgemm_macro_kernel(mc, nc, kc, &C[i+j*ldC], ldC);
            }
        }
    }
}
