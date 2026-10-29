// Probe: C++ compiler.  Uses the C++11 features and headers required by
// ulmBLAS, and the C++ runtime library (a C compiler driver like gcc does
// not link it).
#include <complex>
#include <string>
#include <type_traits>
#include <unordered_map>

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
twice(const T &x)
{
    return x + x;
}

int
main()
{
    std::complex<double>  z(1, 2);
    auto                  y = twice(z.imag());
    static_assert(std::is_same<decltype(y), double>::value, "C++11");

    std::unordered_map<std::string, double> *map
        = new std::unordered_map<std::string, double>();
    (*map)["y"] = y;
    bool ok = map->at("y") == 4;
    delete map;
    return ok ? 0 : 1;
}
