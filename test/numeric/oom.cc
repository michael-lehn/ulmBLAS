//
//  Allocation failure of the packing buffers (MemoryPool): the program is
//  terminated with a message on stderr (abort), no exception is thrown and
//  no null pointer is used.
//
//  Fault injection: this test program defines posix_memalign, which the
//  library (linked statically) uses for its buffers.  It allocates with
//  aligned_alloc, unless a failure is requested: then the k-th call (k =
//  failAt) fails with ENOMEM.  Each case runs in a child process (fork) and
//  calls a routine in a new thread (whose buffer cache is empty, so the
//  routine allocates); the child must end with SIGABRT and the message of
//  ulmBLAS.  The first and the second allocation of a call fail (the
//  second after the first has succeeded).
//
//  POSIX only (fork, waitpid, the library's posix_memalign); not built on
//  Windows, where the library allocates with _aligned_malloc.
//
#if !defined(_WIN32)

#include <test/numeric/blas.h>
#include <test/numeric/testing.h>

#include <ulmblas/ulmblas.h>

#include <atomic>
#include <cerrno>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <signal.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

std::atomic<long> allocCalls(0);
std::atomic<long> failAt(0); // 0: no failure

} // namespace

#if defined(__GLIBC__)
#define OOM_NOEXCEPT noexcept
#else
#define OOM_NOEXCEPT
#endif

extern "C" int
posix_memalign(void **block, std::size_t align, std::size_t size) OOM_NOEXCEPT
{
    long call = ++allocCalls;
    if (failAt > 0 && call == failAt) {
	return ENOMEM;
    }
    // aligned_alloc: size a multiple of the alignment
    std::size_t rounded = (size + align - 1) / align * align;
    if (rounded < size) {
	return ENOMEM;
    }
    void *p = aligned_alloc(align, rounded);
    if (!p) {
	return ENOMEM;
    }
    *block = p;
    return 0;
}

namespace {

using namespace test;

struct ChildResult
{
    bool aborted;
    bool exited;
    std::string err;
};

//
//  Runs f in a new thread of a child process in which the k-th call of
//  posix_memalign fails (k = 0: none fails).
//
template <typename F>
ChildResult
inChild(long k, F f)
{
    ChildResult result = {false, false, std::string()};
    int fd[2];

    std::fflush(stdout);
    std::fflush(stderr);
    if (pipe(fd) != 0) {
	result.err = "pipe failed";
	return result;
    }
    pid_t pid = fork();
    if (pid == 0) {
	close(fd[0]);
	dup2(fd[1], 2);
	allocCalls = 0;
	failAt = k;
	std::thread t(f);
	t.join();
	_exit(0);
    }
    close(fd[1]);
    char buffer[256];
    ssize_t len;
    while ((len = read(fd[0], buffer, sizeof(buffer))) > 0) {
	result.err.append(buffer, std::size_t(len));
    }
    close(fd[0]);
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) {
	result.err += "fork or waitpid failed";
	return result;
    }
    result.aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
    result.exited = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    return result;
}

bool
terminates(Context &ctx, const char *what, long k,
           const std::function<void()> &f)
{
    ChildResult r = inChild(k, f);
    return ctx.check(
        r.aborted && r.err.find("ulmBLAS: out of memory") != std::string::npos,
        "%s, allocation %ld fails: %s, stderr: %s", what, k,
        r.aborted ? "aborted" : (r.exited ? "returned" : "other termination"),
        r.err.c_str());
}

template <typename T> struct Operands
{
    explicit Operands(int n)
        : n(n), A(n * n, T(1)), B(n * n, T(1)), C(n * n, T(0)), x(2 * n, T(1)),
          y(2 * n, T(1))
    {
	for (int i = 0; i < n; ++i) {
	    A[i * n + i] = T(n);
	}
    }

    int n;
    std::vector<T> A, B, C, x, y;
};

const int n = 300;

//
//  Calls of the level 3 routines and of the level 2 routines with packed
//  operands.
//
std::vector<std::pair<const char *, std::function<void()>>>
cases()
{
    using cd = std::complex<double>;
    using cf = std::complex<float>;
    static Operands<double> d(n);
    static Operands<float> s(n);
    static Operands<cd> z(n);
    static Operands<cf> c(n);
    static const double one = 1;
    static const cd zone = 1;
    static const cf cone = 1;
    std::vector<std::pair<const char *, std::function<void()>>> v;

    v.push_back({"dgemm", [] {
	             cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n,
		                 n, n, 1, d.A.data(), n, d.B.data(), n, 0,
		                 d.C.data(), n);
                 }});
    v.push_back({"sgemm", [] {
	             cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n,
		                 n, n, 1, s.A.data(), n, s.B.data(), n, 0,
		                 s.C.data(), n);
                 }});
    v.push_back({"zgemm", [] {
	             cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n,
		                 n, n, blas::mp(&zone), blas::mp(z.A.data()), n,
		                 blas::mp(z.B.data()), n, blas::mp(&zone),
		                 blas::mp(z.C.data()), n);
                 }});
    v.push_back({"cgemm", [] {
	             cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n,
		                 n, n, blas::mp(&cone), blas::mp(c.A.data()), n,
		                 blas::mp(c.B.data()), n, blas::mp(&cone),
		                 blas::mp(c.C.data()), n);
                 }});
    v.push_back({"dtrsm (lower)", [] {
	             cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower,
		                 CblasNoTrans, CblasNonUnit, n, n, 1,
		                 d.A.data(), n, d.C.data(), n);
                 }});
    v.push_back({"dtrsm (upper)", [] {
	             cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper,
		                 CblasNoTrans, CblasNonUnit, n, n, 1,
		                 d.A.data(), n, d.C.data(), n);
                 }});
    v.push_back({"dtrmm (lower)", [] {
	             cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower,
		                 CblasNoTrans, CblasNonUnit, n, n, 1,
		                 d.A.data(), n, d.C.data(), n);
                 }});
    v.push_back({"dtrmm (upper)", [] {
	             cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper,
		                 CblasNoTrans, CblasNonUnit, n, n, 1,
		                 d.A.data(), n, d.C.data(), n);
                 }});
    v.push_back({"dsymm", [] {
	             cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, n, n, 1,
		                 d.A.data(), n, d.B.data(), n, 0, d.C.data(),
		                 n);
                 }});
    v.push_back({"zhemm", [] {
	             cblas_zhemm(CblasColMajor, CblasLeft, CblasLower, n, n,
		                 blas::mp(&zone), blas::mp(z.A.data()), n,
		                 blas::mp(z.B.data()), n, blas::mp(&zone),
		                 blas::mp(z.C.data()), n);
                 }});
    v.push_back({"dsyrk", [] {
	             cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, n,
		                 1, d.A.data(), n, 0, d.C.data(), n);
                 }});
    v.push_back({"zherk", [] {
	             cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans, n, n,
		                 1, blas::mp(z.A.data()), n, 0,
		                 blas::mp(z.C.data()), n);
                 }});
    v.push_back({"dsyr2k", [] {
	             cblas_dsyr2k(CblasColMajor, CblasLower, CblasNoTrans, n, n,
		                  1, d.A.data(), n, d.B.data(), n, 0,
		                  d.C.data(), n);
                 }});
    v.push_back({"zher2k", [] {
	             cblas_zher2k(CblasColMajor, CblasUpper, CblasNoTrans, n, n,
		                  blas::mp(&zone), blas::mp(z.A.data()), n,
		                  blas::mp(z.B.data()), n, 0,
		                  blas::mp(z.C.data()), n);
                 }});
    //  the BLAS interfaces always pass a row or column major A, for which
    //  gemv does not pack: C++ interface, A with general strides
    v.push_back({"gemv (general strides)", [] {
	             ulmBLAS::gemv<std::ptrdiff_t>(
	                 n / 2, n / 2, one, d.A.data(), 2, 2 * n, d.x.data(), 1,
	                 one, d.y.data(), 1);
                 }});
    v.push_back({"dger (increments 2)", [] {
	             cblas_dger(CblasColMajor, n, n, one, d.x.data(), 2,
		                d.y.data(), 2, d.C.data(), n);
                 }});
    return v;
}

template <typename T>
void
hugeBlock()
{
    static ulmBLAS::MemoryPool<T> pool;
    T *p = pool.allocate(std::size_t(1) << 57);
    p[0] = T(1); // not reached
}

template <typename T>
void
overflowingBlock()
{
    static ulmBLAS::MemoryPool<T> pool;
    T *p = pool.allocate(SIZE_MAX / sizeof(T) + 1);
    p[0] = T(1); // not reached
}

} // namespace

ULMBLAS_TEST(T0357,
             "auxiliary/MemoryPool: a block that cannot be allocated "
             "(2^57 elements, size overflows size_t) terminates the program "
             "with a message")
{
    bool ok =
        terminates(ctx, "2^57 doubles", 0, hugeBlock<double>) &&
        terminates(ctx, "2^57 complex<double>", 0,
	           hugeBlock<std::complex<double>>) &&
        terminates(ctx, "SIZE_MAX/8+1 doubles", 0, overflowingBlock<double>) &&
        terminates(ctx, "SIZE_MAX/8+1 floats", 0, overflowingBlock<float>);
    // a block of usual size in a child: no termination
    ChildResult r = inChild(0, [] {
	static ulmBLAS::MemoryPool<double> pool;
	double *p = pool.allocate(1000);
	p[999] = 1;
	pool.release(p);
    });
    ok = ok &&
         ctx.check(r.exited, "allocate(1000) in a child: %s", r.err.c_str());
    if (ok) {
	ctx.pass();
    }
}

ULMBLAS_TEST(T0358,
             "interface/out of memory: level 3 routines and packing "
             "level 2 routines terminate with a message if a buffer cannot "
             "be allocated (fault injection, first and second allocation)")
{
    bool ok = true;
    for (auto &c : cases()) {
	// without failure: the call allocates and returns
	ChildResult r = inChild(0, c.second);
	ok = ok && ctx.check(r.exited, "%s without failure: %s", c.first,
	                     r.err.c_str());
	ok = ok && terminates(ctx, c.first, 1, c.second);
    }
    // the second allocation of a call (the first one succeeded)
    for (auto &c : cases()) {
	if (std::string(c.first).find("gemv") == std::string::npos &&
	    std::string(c.first).find("ger") == std::string::npos) {
	    ok = ok && terminates(ctx, c.first, 2, c.second);
	}
    }
    if (ok) {
	ctx.pass();
    }
}

#endif // !defined(_WIN32)
