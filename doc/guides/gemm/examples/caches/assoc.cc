// Associativity: a cyclic chain of pointers through N cache lines that are
// a fixed stride apart.  If the stride is a multiple of (number of sets) x
// (line size), all lines fall into the same set of the cache; as soon as N
// exceeds the number of ways, they evict each other, although the cache as
// a whole is nearly empty.  With one more line added to the stride, the
// lines spread over different sets.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

double
nsPerLoad(std::size_t lines, std::size_t stride)
{
    std::vector<char> memory(lines * stride + 4096);
    std::vector<void **> slot(lines);
    for (std::size_t i = 0; i < lines; ++i) {
	slot[i] = reinterpret_cast<void **>(&memory[i * stride]);
    }
    for (std::size_t i = 0; i < lines; ++i) { // a cycle through the lines
	*slot[i] = slot[(i + 1) % lines];
    }
    const long loads = 20000000;
    void **p = slot[0];
    for (std::size_t i = 0; i < 4 * lines; ++i) {
	p = static_cast<void **>(*p);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < loads; ++i) {
	p = static_cast<void **>(*p);
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (p == nullptr) {
	std::printf("?");
    }
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / loads;
}

int
main(int argc, char **argv)
{
    // the size of one way of the L1 cache: sets x line size
    const std::size_t way = argc > 1 ? std::atol(argv[1]) : 16384;
    const std::size_t line = argc > 2 ? std::atol(argv[2]) : 128;
    std::printf("stride %zu (one way of L1) and %zu (one line more), "
                "ns/load\n",
                way, way + line);
    std::printf("%6s %10s %10s\n", "lines", "same set", "spread");
    for (std::size_t n = 2; n <= 24; n += 2) {
	std::printf("%6zu %10.2f %10.2f\n", n, nsPerLoad(n, way),
	            nsPerLoad(n, way + line));
    }
}
