// The type of a computation, from the types of its operands: decltype of
// an expression with zeros of these types, as in ulmBLAS
//     typedef decltype(Alpha(0) * TA(0) * TB(0)) T;
#include <complex>
#include <cstdio>

// the name of a type, by full specialization
template <typename T> struct Name;
template <> struct Name<float>
{
    static const char *
    get()
    {
	return "float";
    }
};
template <> struct Name<double>
{
    static const char *
    get()
    {
	return "double";
    }
};
template <> struct Name<std::complex<float>>
{
    static const char *
    get()
    {
	return "std::complex<float>";
    }
};
template <> struct Name<std::complex<double>>
{
    static const char *
    get()
    {
	return "std::complex<double>";
    }
};

template <typename Alpha, typename TA, typename TB>
void
show(const char *what)
{
    typedef decltype(Alpha(0) * TA(0) * TB(0)) T;
    std::printf("%-40s T = %s\n", what, Name<T>::get());
}

int
main()
{
    typedef std::complex<float> fcomplex;
    typedef std::complex<double> dcomplex;

    show<float, float, float>("alpha float,  A float,    B float");
    show<double, float, float>("alpha double, A float,    B float");
    show<float, double, double>("alpha float,  A double,   B double");
    show<float, fcomplex, fcomplex>("alpha float,  A fcomplex, B fcomplex");
    show<dcomplex, double, double>("alpha dcomplex, A double, B double");
#ifdef MIXED
    // std::complex<float> * double: no such operator
    show<double, fcomplex, fcomplex>("alpha double, A fcomplex, B fcomplex");
#endif
}
