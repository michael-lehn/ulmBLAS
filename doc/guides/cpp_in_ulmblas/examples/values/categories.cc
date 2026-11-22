// The value category of an expression e, made visible with decltype((e)):
// T& for an lvalue, T&& for an xvalue, T for a prvalue.
#include <cstdio>
#include <utility>

template <typename T> struct Category
{
    static const char *
    name()
    {
	return "prvalue";
    }
};

template <typename T> struct Category<T &>
{
    static const char *
    name()
    {
	return "lvalue";
    }
};

template <typename T> struct Category<T &&>
{
    static const char *
    name()
    {
	return "xvalue";
    }
};

#define SHOW(e) std::printf("%-26s %s\n", #e, Category<decltype((e))>::name())

int x = 1;
int a[3] = {1, 2, 3};

int &
ref()
{
    return x;
}

int
val()
{
    return x;
}

int &&
rref()
{
    return static_cast<int &&>(x);
}

struct S
{
    int m;
};

int
main()
{
    int *p = a;
    S s = {1};

    SHOW(x);
    SHOW(42);
    SHOW(x + 1);
    SHOW(a[1]);
    SHOW(*p);
    SHOW(p + 1);
    SHOW("abc");
    SHOW(++x);
    SHOW(x++);
    SHOW(x = 2);
    SHOW(x > 0 ? x : a[0]);
    SHOW(x > 0 ? x : 0);
    SHOW(ref());
    SHOW(val());
    SHOW(rref());
    SHOW(std::move(x));
    SHOW(static_cast<int &&>(x));
    SHOW(s.m);
    SHOW(S().m);
    SHOW(std::move(s).m);
    SHOW(double(x));
}
