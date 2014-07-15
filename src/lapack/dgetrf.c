#include <float.h>
#include <math.h>
#include <stdio.h>

#include <ulmblas.h>
#include <auxiliary/xerbla.h>

static void
dscal(const int    n,
      const double alpha,
      double       *x,
      const int    incX)
{
    int    i;

    if (n<=0 || incX<=0) {
        return;
    }
    for (i=0; i<n; ++i) {
        x[i*incX] *= alpha;
    }
}


static void
dswap(const int n,
      double    *x,
      const int incX,
      double    *y,
      const int incY)
{
    int    i;
    double tmp;

    if (n==0) {
        return;
    }
    for (i=0; i<n; ++i) {
        tmp       = x[i*incX];
        x[i*incX] = y[i*incY];
        y[i*incY] = tmp;
    }
}

static int
idamax(const int       n,
       const double    *x,
       const int       incX)
{
    int    i, iamax = 0;
    double amax;

    if (n<1 || incX<=0) {
        return 0;
    }
    if (n==1) {
        return 1;
    }
    iamax = 0;
    amax  = fabs(x[0]);
    for (i=1; i<n; ++i) {
        if (fabs(x[i*incX])>amax) {
            iamax = i;
            amax  = fabs(x[i*incX]);
        }
    }
    return iamax;
}


static void
dger(int            m,
     int            n,
     double         alpha,
     const double   *x,
     int            incX,
     const double   *y,
     int            incY,
     double         *C,
     int            incRowC,
     int            incColC)
{
    int i, j;

    if (m==0 || n==0 || alpha==0.0) {
        return;
    }

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            C[i*incRowC+j*incColC] += alpha*x[i*incX]*y[j*incY];
        }
    }
}

static double
safeMin()
{
//
//  Assume rounding, not chopping. Always.
//
    const double eps   = 0.5*DBL_EPSILON;
    const double small = 1.0 / DBL_MAX;

    double safeMin = DBL_MIN;

    if (small>=safeMin) {
//
//      Use SMALL plus a bit, to avoid the possibility of rounding
//      causing overflow when computing  1/sfmin.
//
        safeMin = small*(1.0+eps);
    }
    return safeMin;
}

void printMatrix();

static int
dgetf2(int      m,
       int      n,
       double   *A,
       int      incRowA,
       int      incColA,
       int      *piv)
{
    int i, j, jp, info;

    const double sfMin = safeMin();

    info = -1;

    if (m==0 || n==0) {
        return info;
    }

    for (j=0; j+1<m && j+1<n; ++j) {
//
//      Find pivot and test for singularity.
//
        jp = j + idamax(m-j, &A[j*incRowA+j*incColA], incRowA);
        piv[j] = jp;

        if (A[jp*incRowA+j*incColA]!=0.0) {
//
//          Apply the interchange to columns 1:N
//
            if (jp!=j) {
                dswap(n, &A[j*incRowA], incColA, &A[jp*incRowA], incColA);
            }
//
//          Compute elements J+1:M of J-th column
//
            if (j+1<m) {
                if (fabs(A[j*incRowA+j*incColA])>=sfMin) {
                    dscal(m-j-1, 1.0/A[j*incRowA+j*incColA],
                          &A[(j+1)*incRowA+j*incColA], incRowA);
                } else {
                    for (i=0; i<m-j-1; ++i) {
                        A[(j+i)*incRowA+j*incColA] /= A[j*incRowA+j*incColA];
                    }
                }
            }
        } else if (info==-1) {
            info = j;
        }
        if (j+1<m && j+1<n) {
//
//          Update trailing submatrix
//
            dger(m-j-1, n-j-1, -1.0,
                 &A[(j+1)*incRowA+ j   *incColA], incRowA,
                 &A[ j   *incRowA+(j+1)*incColA], incColA,
                 &A[(j+1)*incRowA+(j+1)*incColA], incRowA, incColA);
        }
    }
    return info;
}

int
ULMBLAS(dgetrf)(int      m,
                int      n,
                double   *A,
                int      ldA,
                int      *piv)
{
    return dgetf2(m, n, A, 1, ldA, piv) + 1;
}
