//
//  gemm.cc: C = alpha*A*B + beta*C with the C++ interface of ulmBLAS.
//
//  The C++ interface is header only: no library is needed.  Matrices are
//  given by a pointer and two strides (incRow, incCol), so column major,
//  row major and transposed views are all the same call.  The element type
//  is a template parameter: for types other than float, double,
//  std::complex<float> and std::complex<double> (here long double) the
//  block sizes are declared by a specialization of ulmBLAS::BlockSize.
//
//  Build (in the top-level directory of ulmBLAS):
//
//      c++ -std=c++11 -O3 -I. examples/gemm.cc -o gemm
//      ./gemm
//
//  (with ARCHFLAGS, e.g. -mavx -mfma or -mcpu=apple-m4, the optimized micro
//  kernels are used as in the library, see make check_arch)
//
#include <cstdio>
#include <ulmblas/ulmblas.h>

//  block sizes for long double: cache blocks MC x KC (A) and KC x NC (B),
//  register tile MR x NR of the portable micro kernel
namespace ulmBLAS {

template <>
struct BlockSize<long double>
{
    static constexpr int MC = 96, KC = 256, NC = 1024, MR = 4, NR = 4;
    static constexpr int PACK_A_PANELS = 4, PACK_A_GROUP_DISTANCE = 8192;
};

} // namespace ulmBLAS

int
main()
{
    //  column major: A is 2 x 3, B is 3 x 2, C is 2 x 2
    double A[] = { 1, 4,    2, 5,    3, 6 };
    double B[] = { 1, 0, 2,    0, 1, 1 };
    double C[] = { 0, 0,    0, 0 };

    //  C = 1.0*A*B + 0.0*C; the strides of a column major m x n matrix
    //  with leading dimension ld are incRow = 1, incCol = ld
    ulmBLAS::gemm(2L, 2L, 3L,
                  1.0,
                  false, A, 1L, 2L,         // conjA, A, incRowA, incColA
                  false, B, 1L, 3L,         // conjB, B, incRowB, incColB
                  0.0,
                  C, 1L, 2L);               // C, incRowC, incColC
    std::printf("C = [ %g %g ; %g %g ]\n", C[0], C[2], C[1], C[3]);

    //  the same product in long double, B^T read as a row major matrix
    //  (swapped strides): D = A*(B^T)^T = A*B
    long double Bt[] = { 1, 0,    0, 1,    2, 1 };   // B^T, 2 x 3, column major
    long double D[]  = { 0, 0,    0, 0 };
    long double Al[] = { 1, 4,    2, 5,    3, 6 };

    ulmBLAS::gemm(2L, 2L, 3L,
                  1.0L,
                  false, Al, 1L, 2L,
                  false, Bt, 2L, 1L,        // B = (B^T)^T: incRow 2, incCol 1
                  0.0L,
                  D, 1L, 2L);
    std::printf("D = [ %Lg %Lg ; %Lg %Lg ]\n", D[0], D[2], D[1], D[3]);

    //  expected: C = D = [ 7 5 ; 16 11 ]
    bool ok = C[0]==7 && C[1]==16 && C[2]==5 && C[3]==11
           && D[0]==7 && D[1]==16 && D[2]==5 && D[3]==11;
    return ok ? 0 : 1;
}
