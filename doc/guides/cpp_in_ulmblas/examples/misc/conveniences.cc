// Small things of C++11 that ulmBLAS uses all over the place.
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <limits>

void
f(int)
{
    std::printf("  f(int)\n");
}

void
f(const char *)
{
    std::printf("  f(const char *)\n");
}

enum class Platform
{
    Generic,
    Neon,
    Sme
};

const char *
name(Platform p)
{
    switch (p) {
    case Platform::Generic:
	return "generic";
    case Platform::Neon:
	return "neon";
    case Platform::Sme:
	return "sme";
    }
    return "?";
}

int
main()
{
    // 1. nullptr: a null pointer that is not an integer
    std::printf("f(0):\n");
    f(0);
    std::printf("f(nullptr):\n");
    f(nullptr);

    // 2. std::min and std::max: both arguments must have the same type
    std::ptrdiff_t n = 1000;
    const int nb = 256;
    std::ptrdiff_t b = std::min(n, std::ptrdiff_t(nb)); // not min(n, nb)
    std::printf("min = %td\n", b);
#ifdef WRONG
    b = std::min(n, nb); // std::ptrdiff_t and int: no common T
#endif

    // 3. limits of a type
    std::printf("eps = %g, largest ptrdiff_t = %td\n",
                std::numeric_limits<double>::epsilon(),
                std::numeric_limits<std::ptrdiff_t>::max());

    // 4. an enumeration with its own scope, no conversion to int
    const Platform p = Platform::Neon;
    std::printf("platform = %s\n", name(p));

    // 5. auto and the range-based for loop
    const double x[3] = {1, 2, 3};
    double s = 0;
    for (auto v : x) {
	s += v;
    }
    std::printf("sum = %g\n", s);
}
