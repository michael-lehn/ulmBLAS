// Computing with types: a trait by partial specialization, a choice of
// overloads with std::enable_if, and a check with static_assert.
#include <complex>
#include <cstdio>
#include <type_traits>

// 1. a trait: IsComplex<T>::value is true for std::complex<R> only
template <typename T> struct IsComplex : std::false_type
{
};

template <typename R> struct IsComplex<std::complex<R>> : std::true_type
{
};

// 2. two overloads of conjugate; for each T exactly one is viable
template <typename T>
typename std::enable_if<!IsComplex<T>::value, T>::type
conjugate(const T &x)
{
    return x; // real: nothing to do
}

template <typename T>
typename std::enable_if<IsComplex<T>::value, T>::type
conjugate(const T &x)
{
    return T(x.real(), -x.imag());
}

// 3. a requirement that is checked at compile time
template <typename T>
T
dotc(int n, const T *x, const T *y)
{
    static_assert(std::is_floating_point<T>::value || IsComplex<T>::value,
                  "dotc: T must be a real or complex floating-point type");
    T s = T(0);
    for (int i = 0; i < n; ++i) {
	s += conjugate(x[i]) * y[i];
    }
    return s;
}

int
main()
{
    static_assert(IsComplex<std::complex<float>>::value, "");
    static_assert(!IsComplex<double>::value, "");

    const double x[2] = {1, 2}, y[2] = {3, 4};
    const std::complex<double> u[2] = {{1, 1}, {0, 2}};
    const std::complex<double> v[2] = {{1, 0}, {0, 1}};
    std::printf("dotc real:    %g\n", dotc(2, x, y));
    const std::complex<double> s = dotc(2, u, v);
    std::printf("dotc complex: (%g,%g)\n", s.real(), s.imag());
#ifdef WRONG
    const int k[2] = {1, 2};
    dotc(2, k, k);
#endif
}
