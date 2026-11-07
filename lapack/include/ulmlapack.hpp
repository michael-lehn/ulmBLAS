#ifndef ULMLAPACK_HPP
#define ULMLAPACK_HPP 1

//
//  C++ interface: like the C interface (ulmlapack.h), but for matrices
//  with an arbitrary row and column stride, as everywhere in ulmBLAS:
//  A(i,j) = A[i*incRowA + j*incColA].  Column-major with leading dimension
//  ldA is (1, ldA), row-major is (ldA, 1), every second row of a
//  column-major matrix is (2, ldA), and so on.  The elements of A must not
//  overlap; the strides are not checked.
//
//  Same conventions as ulmBLAS::getrf: piv is 0-based (row i was
//  interchanged with row piv[i*incPiv]), the return value is INFO (0, or
//  j+1 for the first exactly zero pivot U(j,j) resp. the first leading
//  minor of order j+1 that is not positive definite).
//

#include <cstddef>
#include <vector>

#include <ulmlapack.h>

namespace ulmlapack {

typedef std::ptrdiff_t Index;

Index
getrf(Index m, Index n, double *A, Index incRowA, Index incColA,
      Index *piv, Index incPiv = 1, const ulmlapack_param *param = 0);

//  Lower: A = L L^T, upper: A = U^T U; only that triangle is referenced.
Index
potrf(bool lower, Index n, double *A, Index incRowA, Index incColA,
      const ulmlapack_param *param = 0);

//
//  The same, but the task graph works on a copy: a contiguous column-major
//  one (_colmajor), or the tile layout (_tiled: every nb x nb tile stored
//  contiguously).  The copy in and back is done by tasks of the same graph
//  (one per tile), so it overlaps with the factorization.  The workspace is
//  kept per calling thread and reused by later calls.
//
Index
getrf_colmajor(Index m, Index n, double *A, Index incRowA, Index incColA,
               Index *piv, Index incPiv = 1, const ulmlapack_param *param = 0);

Index
potrf_colmajor(bool lower, Index n, double *A, Index incRowA, Index incColA,
               const ulmlapack_param *param = 0);

Index
potrf_tiled(bool lower, Index n, double *A, Index incRowA, Index incColA,
            const ulmlapack_param *param = 0);

//
//  A matrix in tile layout, for computations that stay in it: tile (i,j) is
//  nb x nb, column major (leading dimension nb), stored at
//  tile(0,0) + (i + j*mt)*nb*nb.  Edge tiles are stored with full size.
//
class TileMatrix
{
    public:
        TileMatrix(Index m_, Index n_, Index nb_)
            : m(m_), n(n_), nb(nb_), mt((m_+nb_-1)/nb_), nt((n_+nb_-1)/nb_),
              data_(std::size_t(mt)*nt*nb_*nb_)
        {
        }

        double *
        tile(Index i, Index j)
        {
            return &data_[std::size_t(i+j*mt)*nb*nb];
        }

        double &
        operator()(Index i, Index j)
        {
            return tile(i/nb, j/nb)[i%nb + (j%nb)*nb];
        }

        //  copy in from / out to a strided matrix
        void
        assign(const double *A, Index incRowA, Index incColA)
        {
            for (Index j=0; j<n; ++j) {
                for (Index i=0; i<m; ++i) {
                    (*this)(i, j) = A[i*incRowA+j*incColA];
                }
            }
        }

        void
        copyTo(double *A, Index incRowA, Index incColA)
        {
            for (Index j=0; j<n; ++j) {
                for (Index i=0; i<m; ++i) {
                    A[i*incRowA+j*incColA] = (*this)(i, j);
                }
            }
        }

        const Index m, n, nb, mt, nt;

    private:
        std::vector<double> data_;
};

//  Cholesky factorization of the lower triangle of a square TileMatrix
//  (A = L L^T, tile size A.nb; the other parameters as above).
Index
potrf(TileMatrix &A, const ulmlapack_param *param = 0);

} // namespace ulmlapack

#endif // ULMLAPACK_HPP
