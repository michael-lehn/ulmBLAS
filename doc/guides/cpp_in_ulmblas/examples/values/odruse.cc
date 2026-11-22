// A static constexpr data member that is bound to a reference (here by
// std::min, which takes const long &) is odr-used: in C++11 and C++14 it
// then needs a definition outside the class.  Without -DFIXED the program
// does not link with -std=c++11 -O0; run.sh keeps the message.
#include <algorithm>
#include <cstdio>

struct Kernel
{
    static constexpr long MR = 4;
};

int
main(int argc, char **)
{
    long m = 10 + argc;
#ifndef FIXED
    long mr = std::min(m, Kernel::MR); // binds Kernel::MR to const long &
#else
    long mr = std::min(m, long(Kernel::MR)); // binds a prvalue: a copy
#endif
    std::printf("mr = %ld\n", mr);
}
