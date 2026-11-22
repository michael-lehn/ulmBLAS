// Template argument deduction: every argument that mentions IndexType must
// give the same type.  Compiling this file fails on purpose (run.sh keeps
// the first lines of the messages); with -DFIXED it compiles.
#include <cstddef>

template <typename IndexType, typename Alpha, typename VX>
void
scal(IndexType n, const Alpha &alpha, VX *x, IndexType incX)
{
    for (IndexType i = 0; i < n; ++i) {
	x[i * incX] *= alpha;
    }
}

int
main()
{
    double x[4] = {1, 2, 3, 4};
    long n = 4;
#ifndef FIXED
    scal(n, 2.0, x, 1); // IndexType: long from n, int from 1
#else
    scal<std::ptrdiff_t>(n, 2.0, x, 1); // IndexType given, the rest deduced
#endif
}
