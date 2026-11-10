#ifndef ULMBLAS_LAPACK_TASKS_POTRFTASKS_H
#define ULMBLAS_LAPACK_TASKS_POTRFTASKS_H 1

namespace ulmBLAS {
namespace tasks {

//
//  Tiled Cholesky factorization A = L L^H of the lower triangle with
//  OpenMP tasks and dependencies (right-looking, as PLASMA's pzpotrf), in
//  place, any strides.  nb: tile size, threads: threads of the parallel
//  region.  Returns 0 or i if the leading minor of order i is not positive
//  definite.  Without OpenMP the tasks run in the order they are created,
//  which is a valid sequential order.
//
template <typename IndexType, typename T>
IndexType
potrfTasks(IndexType n, T *A, IndexType incRowA, IndexType incColA,
           IndexType nb, int threads);

} // namespace tasks
} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_POTRFTASKS_H

#include <ulmblas/lapack/tasks/potrftasks.tcc>
