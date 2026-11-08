#ifndef ULMLAPACK_TILES_H
#define ULMLAPACK_TILES_H 1

//
//  How the task graphs address their tiles.
//
//  - strided: tile (i,j) is the submatrix A[i*nb*rs + j*nb*cs] with strides
//    (rs, cs) of the matrix itself (in place) or of a column-major copy;
//  - tiled: every tile is stored contiguously, nb x nb column major, tile
//    (i,j) at A + (i + j*mt)*nb*nb, strides (1, nb).
//
//  Either way a tile is a pointer and two strides, so the kernels do not
//  care.  The LU panel needs the tiles of a block column to form one
//  strided matrix: only the strided form.
//

#include <cstddef>
#include <memory>

#include "kernels.h"

namespace ulmlapack {

struct Tiles
{
    double  *A;
    Index   rs, cs;
    Index   nb;
    Index   mt;         // tiled: tile rows
    bool    tiled;

    double *
    tile(Index i, Index j) const
    {
        return tiled ? A + (i + j*mt)*nb*nb
                     : A + i*nb*rs + j*nb*cs;
    }
};

inline Tiles
stridedTiles(double *A, Index rs, Index cs, Index nb)
{
    Tiles t = {A, rs, cs, nb, 0, false};
    return t;
}

inline Tiles
tiledTiles(double *A, Index mt, Index nb)
{
    Tiles t = {A, 1, nb, nb, mt, true};
    return t;
}

//
//  The caller's matrix, if the task graph works on a copy (copied in and
//  back by tasks of the same graph).
//
struct Source
{
    double  *A;
    Index   rs, cs;
};

//  B = A for an m x n block; lower: only the lower triangle (i >= j)
inline void
copyBlock(Index m, Index n, const double *A, Index ars, Index acs,
          double *B, Index brs, Index bcs, bool lower)
{
    for (Index j=0; j<n; ++j) {
        for (Index i=lower ? j : 0; i<m; ++i) {
            B[i*brs+j*bcs] = A[i*ars+j*acs];
        }
    }
}

//
//  Workspace of the copying variants (slot 0) and of the packed tiles (slot
//  1), kept per calling thread and reused,
//  so that repeated calls do not pay for allocation and page faults again.
//  Not initialized: the copy tasks touch it first (in parallel).
//
inline double *
workspace(std::size_t n, int slot = 0)
{
    thread_local std::unique_ptr<double[]> w[2];
    thread_local std::size_t               size[2] = {0, 0};
    if (size[slot]<n) {
        w[slot].reset(new double[n]);
        size[slot] = n;
    }
    return w[slot].get();
}

} // namespace ulmlapack

#endif // ULMLAPACK_TILES_H
