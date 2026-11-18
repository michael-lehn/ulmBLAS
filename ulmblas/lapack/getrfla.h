#ifndef ULMBLAS_LAPACK_GETRFLA_H
#define ULMBLAS_LAPACK_GETRFLA_H 1

#include <cstddef>

namespace ulmBLAS {

//
//  LU factorization with partial pivoting (as getrf), blocked right-looking
//  with a lookahead of one panel on nt threads of the pool.  Step K (panel K
//  factored):
//
//      thread 0     interchanges of panel K in the next block column and
//                   its triangular solve
//      all threads  the rows of the next block column updated with panel K
//                   (gemm, rows split)
//      thread 0     the next panel factored (getrf2, serial), then as the
//                   others:
//      all threads  the columns right of it updated with panel K in chunks
//                   of columns (interchanges, trsm, gemm of a chunk on one
//                   thread; guided: a share of the remaining columns)
//
//  One barrier per step.  The interchanges left of each panel follow at
//  the end, panel by panel on all threads.
//
//  nb: panel width; nbMin > 0: adaptive widths between nbMin and nb (see
//  Steps::width); chunk: minimal columns of a chunk (0: 64).  No work
//  space.
//
template <typename IndexType, typename T, typename PivIndex>
IndexType
getrfLookahead(IndexType m, IndexType n, T *A, IndexType incRowA,
               IndexType incColA, PivIndex *piv, IndexType incPiv, IndexType nb,
               IndexType nbMin, IndexType chunk, std::ptrdiff_t nt);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRFLA_H

#include <ulmblas/lapack/getrfla.tcc>
