#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <cstdio>

extern "C" {

void
F77BLAS(xerbla)(const char                *rout,
                const ULMBLAS_INDEX_TYPE  *info,
                std::size_t               len)
{
    fprintf(stderr, "Parameter %lld to routine %.*s was incorrect\n",
            (long long)*info, int(len), rout);
}

} // extern "C"
