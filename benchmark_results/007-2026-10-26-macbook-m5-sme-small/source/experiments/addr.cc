// addresses returned by new double[] / delete [] in a loop (as gemm with a
// plain allocator: A buffer, B buffer, release both, next call)
#include <cstdio>
#include <cstdlib>
int main(int argc, char **argv)
{
    std::size_t na = std::atol(argv[1]), nb = std::atol(argv[2]);
    for (int c=0; c<8; ++c) {
        double *a = new double[na], *b = new double[nb];
        a[0] = b[0] = 1;
        std::printf("call %d: A %p  B %p\n", c, (void *)a, (void *)b);
        delete [] b; delete [] a;
    }
}
