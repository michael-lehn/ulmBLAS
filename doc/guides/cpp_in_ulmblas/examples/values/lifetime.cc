// A function that returns its const reference parameter, as real() of
// ulmBLAS does for real types.  Used inside an expression it is safe; a
// reference that keeps its result dangles.  run.sh builds this file with
// -fsanitize=address, which reports the use of the dead temporary.
#include <cstdio>

template <typename T>
const T &
real(const T &x)
{
    return x;
}

int
main(int argc, char **)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0); // output before the report
    double a = 1.5, b = argc; // b = 1: not known at compile time

    // 1. inside one expression: the temporary a * b lives until the ';'
    double s = real(a * b) + 1;
    std::printf("s = %g\n", s);

    // 2. a reference bound directly to a temporary extends its lifetime
    const double &r1 = a * b;
    std::printf("r1 = %g\n", r1);

    // 3. bound through a function: the temporary dies at the ';', r2
    //    refers to a dead object
    const double &r2 = real(a * b);
    std::printf("r2 = %g\n", r2);
}
