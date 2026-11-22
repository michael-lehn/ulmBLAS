/*
 *  Hooks of FlexiBLAS (a BLAS and LAPACK wrapper that loads a backend such
 *  as this library at run time): FlexiBLAS looks them up with dlsym, each
 *  is optional.  The types are those of FlexiBLAS (int for the number of
 *  threads and INFO, the LP64 interface).
 *
 *  - flexiblas_backend_set_num_threads: flexiblas_set_num_threads and its
 *    aliases (openblas_set_num_threads, ..., also the Fortran variants: if
 *    the backend has no flexiblas_backend_set_num_threads_, FlexiBLAS calls
 *    this one) set the threads of ulmBLAS, as ulmblas_set_num_threads.
 *  - flexiblas_backend_get_num_threads: the number of threads, as
 *    ulmblas_get_num_threads.  FlexiBLAS looks it up under the name
 *    flexiblas_backend_ge_num_threads (a typo), so flexiblas_get_num_threads
 *    finds it only once that name is corrected.
 *  - __flexiblas_set_xerbla: FlexiBLAS passes the XERBLA of the program, if
 *    it defines one; the errors of this library then reach it also where
 *    calls inside the library are bound to its own xerbla_ (macOS), as
 *    with ulmblas_set_f77_xerbla.
 *
 *  ULMBLAS_FLEXIBLAS_INFO: the type of INFO, int (default, libulmblas and
 *  libflexiblas) or int64_t (libulmblas64 and libflexiblas64).
 */
#include <stddef.h>
#include <stdint.h>

#ifndef ULMBLAS_FLEXIBLAS_INFO
#define ULMBLAS_FLEXIBLAS_INFO int
#endif

typedef void (*ulmblas_f77_xerbla_t)(const char *srname,
                                     const ULMBLAS_FLEXIBLAS_INFO *info,
                                     size_t lenSrname);

void
ulmblas_set_num_threads(int n);

int
ulmblas_get_num_threads(void);

void
ulmblas_set_f77_xerbla(ulmblas_f77_xerbla_t handler);

void
flexiblas_backend_set_num_threads(int n)
{
    ulmblas_set_num_threads(n);
}

int
flexiblas_backend_get_num_threads(void)
{
    return ulmblas_get_num_threads();
}

void
__flexiblas_set_xerbla(ulmblas_f77_xerbla_t handler)
{
    ulmblas_set_f77_xerbla(handler);
}
