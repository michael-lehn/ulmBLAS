#ifndef ULMBLAS_LAPACK_TASKS_SOLVETASKS_H
#define ULMBLAS_LAPACK_TASKS_SOLVETASKS_H 1

namespace ulmBLAS {
namespace tasks {

//
//  A step of a solve with the n x nrhs matrix B (getrs, potrs, trtrs):
//
//      SwapForward   rows i and piv[i] of B are interchanged for i = 0,
//                    ..., n-1 (piv 0-based)
//      SwapBackward  the same for i = n-1, ..., 0
//      Solve         B := op(A)^{-1} B with the triangular n x n matrix A
//                    (upper or lower, conj: conjugated elements, unitDiag:
//                    the diagonal is not referenced)
//
//  A transposed A is passed as a view: strides exchanged, upper <-> lower.
//
template <typename IndexType, typename T> struct SolveStep
{
    enum Kind
    {
	SwapForward,
	SwapBackward,
	Solve
    };

    Kind kind;
    const IndexType *piv; // swaps
    bool upper, conj, unitDiag;
    const T *A; // solve
    IndexType incRowA, incColA;
};

//
//  Runs the steps.  With OpenMP and threads > 1 as one task graph on
//  nb x nb tiles of B (as PLASMA's pztrsm): per tile row k of a solve a
//  TRSM task per tile B(k,j), then GEMM tasks B(i,j) -= A(i,k) B(k,j) for
//  the other tile rows i; a swap is one task per column of tiles.  The
//  steps are pipelined (the next step of a column of tiles starts when its
//  tiles are done).  Otherwise the steps run one after the other with the
//  serial BLAS routines on all of B.
//
template <typename IndexType, typename T>
void
solveSteps(IndexType n, IndexType nrhs, const SolveStep<IndexType, T> *steps,
           int numSteps, T *B, IndexType incRowB, IndexType incColB,
           IndexType nb, int threads);

//
//  solveSteps with the threads and tile size of solveParam (param.h): with
//  OpenMP and enough work as tasks, inside a parallel region and without
//  OpenMP serial.
//
template <typename IndexType, typename T>
void
solve(IndexType n, IndexType nrhs, const SolveStep<IndexType, T> *steps,
      int numSteps, T *B, IndexType incRowB, IndexType incColB);

} // namespace tasks
} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_SOLVETASKS_H

#include <ulmblas/lapack/tasks/solvetasks.tcc>
