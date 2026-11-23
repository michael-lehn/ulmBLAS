#ifndef ULMBLAS_AUXILIARY_CLOCK_H
#define ULMBLAS_AUXILIARY_CLOCK_H 1

//
//  Monotonic time in nanoseconds, from the C library of the system
//  (clock_gettime, QueryPerformanceCounter on Windows) instead of
//  std::chrono::steady_clock: steady_clock::now is a function of the C++
//  runtime, and ulmBLAS links no C++ runtime.
//

#include <cstdint>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

namespace ulmBLAS {

inline std::int64_t
monotonicNanoseconds()
{
#if defined(_WIN32)
    LARGE_INTEGER count, frequency;
    QueryPerformanceCounter(&count);
    QueryPerformanceFrequency(&frequency);
    //  seconds and the rest separately: count * 1e9 would overflow after
    //  some days of uptime
    const std::int64_t f = frequency.QuadPart;
    return count.QuadPart / f * 1000000000 +
           count.QuadPart % f * 1000000000 / f;
#else
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return std::int64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
#endif
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_CLOCK_H
