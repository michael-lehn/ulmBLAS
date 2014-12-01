




#ifndef ULMBLAS_ATLLAPACK_H
#define ULMBLAS_ATLLAPACK_H 1

#ifdef __cplusplus
extern "C" {
#endif

//
//  Constants for Trans, Side, UpLo and Diag are compatible with CBLAS and ATLAS
//
enum Trans {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
    Conj = 114
};

enum Side {
    Left = 141,
    Right = 142
};

enum UpLo {
    Upper = 121,
    Lower = 122
};

enum Diag {
    NonUnit = 131,
    Unit = 132
};

enum Order {
    RowMajor = 101,
    ColMajor = 102
};

//
//  Declaration of BLAS functions currently implemented in ulmBLAS
//

double
ATL_dasum(const int n,
            const double *x,
            const int incX);

void
ATL_daxpy(const int n,
            const double alpha,
            const double *x,
            const int incX,
            double *y,
            int incY);

void
ATL_dcopy(const int n,
            const double *x,
            const int incX,
            double *y,
            const int incY);

void
ATL_dscal(const int n,
            const double alpha,
            double *x,
            const int incX);

void
ATL_dswap(const int n,
            double *x,
            const int incX,
            double *y,
            const int incY);

double
ATL_ddot(const int n,
           const double *x,
           const int incX,
           const double *y,
           const int incY);

int
ATL_idamax(const int n,
             const double *x,
             const int incX);

double
ATL_dnrm2(const int n,
            const double *x,
            const int incX);

void
ATL_drot(const int n,
           double *x,
           const int incX,
           double *y,
           const int incY,
           const double c,
           const double s);

void
ATL_drotg(double *a,
            double *b,
            double *c,
            double *s);

void
ATL_dger(const int m,
           const int n,
           const double alpha,
           const double *x,
           const int incX,
           const double *y,
           const int incY,
           double *A,
           const int ldA);

void
ATL_dgemv(enum Trans transA,
            int m,
            int n,
            double alpha,
            const double *A,
            int ldA,
            const double *x,
            int incX,
            double beta,
            double *y,
            int incY);

void
ATL_dsymv(enum UpLo upLo,
            int n,
            double alpha,
            const double *A,
            int ldA,
            const double *x,
            int incX,
            double beta,
            double *y,
            int incY);

void
ATL_dtrmv(enum UpLo upLo,
            enum Trans trans,
            enum Diag diag,
            int n,
            const double *A,
            int ldA,
            double *x,
            int incX);

void
ATL_dtrsv(enum UpLo upLo,
            enum Trans trans,
            enum Diag diag,
            int n,
            const double *A,
            int ldA,
            double *x,
            int incX);

void
ATL_dgemm(const enum Trans transA,
            const enum Trans transB,
            const int m,
            const int n,
            const int k,
            const double alpha,
            const double *A,
            const int ldA,
            const double *B,
            const int ldB,
            const double beta,
            double *C,
            const int ldC);

void
ATL_dsymm(const enum Side side,
            const enum UpLo upLo,
            const int m,
            const int n,
            const double alpha,
            const double *A,
            const int ldA,
            double *B,
            const int ldB,
            const double beta,
            double *C,
            const int ldC);

void
ATL_dtrmm(const enum Side side,
            const enum UpLo upLo,
            const enum Trans transA,
            const enum Diag diag,
            const int m,
            const int n,
            const double alpha,
            const double *A,
            const int ldA,
            double *B,
            const int ldB);

void
ATL_dtrsm(const enum Side side,
            const enum UpLo upLo,
            const enum Trans transA,
            const enum Diag diag,
            const int m,
            const int n,
            const double alpha,
            const double *A,
            const int ldA,
            double *B,
            const int ldB);

void
ATL_dsyrk(enum UpLo upLo,
            enum Trans trans,
            int n,
            int k,
            double alpha,
            const double *A,
            int ldA,
            double beta,
            double *C,
            int ldC);

void
ATL_dsyr2k(enum UpLo upLo,
             enum Trans trans,
             int n,
             int k,
             double alpha,
             const double *A,
             int ldA,
             const double *B,
             int ldB,
             double beta,
             double *C,
             int ldC);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
