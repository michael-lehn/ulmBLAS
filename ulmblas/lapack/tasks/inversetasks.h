#ifndef ULMBLAS_LAPACK_TASKS_INVERSETASKS_H
#define ULMBLAS_LAPACK_TASKS_INVERSETASKS_H 1

namespace ulmBLAS { namespace tasks {

//
//  Tile algorithms of PLASMA (pztrtri, pzlauum) with OpenMP tasks on nb x nb
//  tiles of A, in place, any strides.  Without OpenMP the tasks run in the
//  order they are created (a valid sequential order).
//
//  trtriLowerTasks:  A := inv(A), A lower triangular (unitDiag: the
//                    diagonal is 1 and not referenced).  A must not be
//                    singular (checked by the caller).
//  lauumUpperTasks:  A := U U^H, U upper triangular (real diagonal).
//
template <typename IndexType, typename T>
    void
    trtriLowerTasks(bool         unitDiag,
                    IndexType    n,
                    T            *A,
                    IndexType    incRowA,
                    IndexType    incColA,
                    IndexType    nb,
                    int          threads);

template <typename IndexType, typename T>
    void
    lauumUpperTasks(IndexType    n,
                    T            *A,
                    IndexType    incRowA,
                    IndexType    incColA,
                    IndexType    nb,
                    int          threads);

} } // namespace tasks, ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_INVERSETASKS_H

#include <ulmblas/lapack/tasks/inversetasks.tcc>
