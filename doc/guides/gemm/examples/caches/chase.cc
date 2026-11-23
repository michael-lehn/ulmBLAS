// Pointer chasing (as in lecture 4 of High Performance Computing I): the
// average time of a load that depends on the previous one, for a cyclic
// chain of pointers in a random order through a memory area of a given
// size.  The times step up where the area no longer fits into a cache.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

double
nsPerLoad(std::size_t bytes)
{
    const std::size_t n = bytes / sizeof(void *);
    std::vector<void *> memory(n);
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) {
	order[i] = i;
    }
    unsigned long state = 1;
    for (std::size_t i = n - 1; i > 0; --i) { // random permutation
	state = state * 6364136223846793005UL + 1442695040888963407UL;
	std::swap(order[i], order[(state >> 33) % (i + 1)]);
    }
    for (std::size_t i = 0; i < n; ++i) { // one cycle through all elements
	memory[order[i]] = &memory[order[(i + 1) % n]];
    }

    const long loads = 20000000;
    void **p = &memory[order[0]];
    for (std::size_t i = 0; i < n; ++i) { // warm up
	p = static_cast<void **>(*p);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < loads; ++i) {
	p = static_cast<void **>(*p);
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (p == nullptr) { // keep the loop
	std::printf("?");
    }
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / loads;
}

int
main()
{
    std::printf("%12s %10s\n", "bytes", "ns/load");
    for (std::size_t kib = 16; kib <= 512 * 1024; kib *= 2) {
	for (std::size_t f : {4, 6}) { // 1 and 1.5 times a power of two
	    const std::size_t bytes = kib * 1024 * f / 4;
	    std::printf("%12zu %10.2f\n", bytes, nsPerLoad(bytes));
	}
    }
}
