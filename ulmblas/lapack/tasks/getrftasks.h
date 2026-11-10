#ifndef ULMBLAS_LAPACK_TASKS_GETRFTASKS_H
#define ULMBLAS_LAPACK_TASKS_GETRFTASKS_H 1

namespace ulmBLAS {
namespace tasks {

//
//  LU factorization with partial pivoting A = P L U of an m x n matrix with
//  OpenMP tasks and dependencies, in place, any strides.  piv: min(m,n)
//  entries, 0-based, contiguous.  nb: tile size, ib: inner block of a
//  panel, panelThreads: max. threads of a panel team (1: the serial
//  recursive algorithm), threads: threads of the parallel region.  Returns
//  0 or i if U(i-1,i-1) is exactly zero (1-based i).  Needs OpenMP.
//
template <typename IndexType, typename T>
IndexType
getrfTasks(IndexType m, IndexType n, T *A, IndexType incRowA, IndexType incColA,
           IndexType *piv, IndexType nb, IndexType ib, int panelThreads,
           int threads);

} // namespace tasks
} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_GETRFTASKS_H

#include <ulmblas/lapack/tasks/getrftasks.tcc>
