#ifndef ULMBLAS_LAPACK_LAENV_TCC
#define ULMBLAS_LAPACK_LAENV_TCC 1

#include <cassert>
#include <cstring>
#include <ulmblas/lapack/laenv.h>

namespace ulmBLAS {

template <typename T>
long
laenv(long        spec,
      const char  *name,
      const char  *opts,
      long        n1,
      long        n2,
      long        n3,
      long        n4)
{
    long result = -1;

    switch (spec) {
//
//      optimal blocksize
//
        case 1:
            result = 32;
            if (strcmp(name, "GETRF")==0) {
                result = 64;
            }
            break;

//
//      minimal blocksize
//
        case 2:
            result = 2;
            break;

//
//      crossover point
//
        case 3:
            result = 32;
            break;

//
//      DEPRECATED: number of shifts used in the nonsymmetric eigenvalue
//                  routines
//
        case 4:
            assert(0);
            break;
//
//      12 <= pec<= 16: hseqr or one of its subroutines ..
//
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
            break;

        default:
            assert(0);
            result = -1;
            break;

    }

    assert(result!=-1);
    return result;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_LAENV_TCC
