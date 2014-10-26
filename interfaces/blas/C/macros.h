#ifndef INTERFACES_BLAS_C_CONFIG_H
#define INTERFACES_BLAS_C_CONFIG_H 1

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
//  Convert diag chars 'u', 'U', 'n', 'N' to corresponding enum Diag constants.
//  Illegal chars result in 0.
//
#define charToDiag(x)  (toupper(x) == 'N' ? NonUnit :     \
                        toupper(x) == 'U' ? Unit : 0)

#endif // INTERFACES_BLAS_C_CONFIG_H 1
