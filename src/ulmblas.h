#ifndef ULMBLAS_H
#define ULMBLAS_H 1

#include <ctype.h>

#define ULM_BLOCKED

#ifdef FAKE_ATLAS
#   define ULMBLAS(x) ATL_##x
#   define F77BLAS(x) x##_intern
#else
#   define ULMBLAS(x) ULM_##x
#   define F77BLAS(x) x##_
#endif

//
//  Constants for Trans, Side and UpLo are compatible with CBLAS and ATLAS
//
enum Trans {
    NoTrans   = 111,
    Trans     = 112,
    ConjTrans = 113,
    Conj      = 114
};

enum Side  {
    Left    = 141,
    Right   = 142
};

enum UpLo  {
    Upper   = 121,
    Lower   = 122
};

//
//  Convert trans chars 'n', 'N', 't', 'T', 'c', 'C', 'r', 'R' to corresponding
//  enum Trans constants.  Illegal chars result in 0.
//
#define charToTranspose(x)  (toupper(x) == 'N' ? NoTrans :   \
                             toupper(x) == 'T' ? Trans :     \
                             toupper(x) == 'C' ? ConjTrans : \
                             toupper(x) == 'R' ? Conj: 0)

//
//  Convert side chars 'l', 'L', 'r', 'R' to corresponding enum Side constants.
//  Illegal chars result in 0.
//
#define charToSide(x)  (toupper(x) == 'L' ? Left :     \
                        toupper(x) == 'R' ? Right : 0)

//
//  Convert uplo chars 'u', 'U', 'l', 'L' to corresponding enum UpLo constants.
//  Illegal chars result in 0.
//
#define charToUpLo(x)  (toupper(x) == 'U' ? Upper :     \
                        toupper(x) == 'L' ? Lower : 0)

//
//  macro for max function
//
#define max(x,y)  (((x)<(y)) ? (y) : (x))

#endif // ULM_BLAS_H
