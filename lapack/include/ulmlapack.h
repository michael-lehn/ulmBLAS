#ifndef ULMLAPACK_H
#define ULMLAPACK_H 1

/*
 *  Task-based LAPACK routines of ulmBLAS (OpenMP tasks with dependencies).
 *
 *  Column-major storage as in LAPACK; the matrix is factored in place
 *  (no conversion to a tile layout).  The return value is LAPACK's INFO:
 *
 *      ulmlapack_dpotrf   0: success, -i: argument i is invalid,
 *                         i > 0: the leading minor of order i is not
 *                         positive (definite), the factorization could not
 *                         be completed
 *      ulmlapack_dgetrf   0: success, -i: argument i is invalid,
 *                         i > 0: U(i,i) is exactly zero; the factorization
 *                         has been completed (as in LAPACK)
 *
 *  ipiv is 1-based as in LAPACK: row i was interchanged with row ipiv[i-1].
 *
 *  The routines start their own OpenMP parallel region (the number of
 *  threads is omp_get_max_threads() unless set in the parameters).  Task
 *  priorities are used for the critical path; they only take effect if the
 *  environment variable OMP_MAX_TASK_PRIORITY is set (e.g. to 3).
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nb;             /* tile size; 0: default (depends on n, threads)    */
    int ib;             /* dgetrf: inner block of the panel; 0: default     */
    int panel_threads;  /* dgetrf: max. threads of a panel; 0: default,
                           1: serial (recursive) panel                      */
    int threads;        /* threads; 0: omp_get_max_threads()                */
    int serial_n;       /* problems with min(m,n) <= serial_n are factored
                           by the serial recursive algorithm in the calling
                           thread; -1: default, 0: never                    */
    int empty_tasks;    /* diagnostics: create and run the task graph with
                           empty tasks (no computation, the result is
                           wrong); measures the cost of the tasks          */
    int prepack;        /* dpotrf: 1: every tile of L is packed once (into
                           the formats of the GEMM micro kernel) after its
                           TRSM, and the GEMM tasks call the macro kernel
                           directly; 0: every GEMM packs its operands       */
    int outer;          /* dpotrf in place: s > 1 groups s tile columns to
                           an outer block; the trailing matrix is updated
                           once per outer block (packed tiles, the update
                           accumulated in a contiguous buffer): s times
                           fewer passes over the matrix, for large strides.
                           0, 1: off                                        */
    int window;         /* with outer: the updates of a trailing tile are
                           deferred until w outer blocks are pending (or
                           its column is next): w times fewer passes, for
                           w+1 blocks of packed tiles; default 1           */
    int strip;          /* with outer: 1: the tiles of the current outer
                           block are factored as contiguous copies (copied
                           in and back by tasks); 0: in place              */
} ulmlapack_param;

/*  Defaults (all fields 0 or -1).                                          */
void ulmlapack_param_init(ulmlapack_param *param);

/*  The parameters actually used for a problem (nb, ib, ... filled in).     */
void ulmlapack_dpotrf_resolve(int n, ulmlapack_param *param);
void ulmlapack_dgetrf_resolve(int m, int n, ulmlapack_param *param);

int ulmlapack_dpotrf(char uplo, int n, double *A, int ldA);
int ulmlapack_dgetrf(int m, int n, double *A, int ldA, int *ipiv);

int ulmlapack_dpotrf_param(char uplo, int n, double *A, int ldA,
                           const ulmlapack_param *param);
int ulmlapack_dgetrf_param(int m, int n, double *A, int ldA, int *ipiv,
                           const ulmlapack_param *param);

/*  Diagnostics of dgetrf since the last call: panels factored inside the
    task algorithm, panels factored by a team of threads, sum of the team
    sizes.  */
void ulmlapack_dgetrf_stats(long *panels, long *teams, long *members);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ULMLAPACK_H */
