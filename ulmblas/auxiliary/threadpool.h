#ifndef ULMBLAS_AUXILIARY_THREADPOOL_H
#define ULMBLAS_AUXILIARY_THREADPOOL_H 1

//
//  The threads of ulmBLAS (no OpenMP: the libraries need no OpenMP
//  runtime).  POSIX threads, Windows threads on Windows; nothing here
//  throws (the interfaces are called from C and Fortran, see
//  interfaces/blas/shared/Makefile): a thread that cannot be started
//  leaves the pool smaller.
//
//  ThreadPool::instance().run(nt, f) calls f(tid, nt) on nt threads, tid =
//  0, ..., nt-1, and returns when all calls have returned; the calling
//  thread is thread 0, the others are workers of the pool, started at the
//  first call and kept waiting afterwards.  The pool runs one parallel
//  region at a time: a call from inside a parallel region (nested) and a
//  call while another thread of the application uses the pool run f(0, 1)
//  on the calling thread.
//
//  The number of threads (size()): set by setSize(n) (ulmblas_set_num_threads),
//  by default ULMBLAS_NUM_THREADS, otherwise OMP_NUM_THREADS, otherwise the
//  CPUs available to the process (Linux: its affinity mask, e.g. from
//  taskset or a batch system; macOS: the performance cores, as parallel
//  regions with static partitions run at the pace of the slowest core).
//  Workers are started when a region needs them.
//
//  Affinity (Linux; threads::Affinity, ulmblas_set_affinity or
//  ULMBLAS_AFFINITY): thread tid runs on the tid-th CPU of the process's
//  affinity mask (one per core first, see threads::CpuOrder).  The calling
//  thread (tid 0) is pinned for the region only and gets its mask back
//  afterwards.  Mode 1: workers keep their CPU until the setting changes;
//  mode 2 (default): workers are pinned for one region and unpinned after
//  it (placed once); mode 0: no pinning.
//
//  Barrier: the threads of a parallel region (or a group of them) wait
//  until all have arrived (spinning, then yielding the processor);
//  trivially destructible.
//

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace ulmBLAS {

namespace threads {

#if defined(_WIN32)

struct Mutex
{
    SRWLOCK lock = SRWLOCK_INIT;

    void
    acquire()
    {
	AcquireSRWLockExclusive(&lock);
    }
    bool
    tryAcquire()
    {
	return TryAcquireSRWLockExclusive(&lock) != 0;
    }
    void
    release()
    {
	ReleaseSRWLockExclusive(&lock);
    }
};

struct Condition
{
    CONDITION_VARIABLE cv = CONDITION_VARIABLE_INIT;

    void
    wait(Mutex &mutex)
    {
	SleepConditionVariableSRW(&cv, &mutex.lock, INFINITE, 0);
    }
    void
    broadcast()
    {
	WakeAllConditionVariable(&cv);
    }
};

inline void
yield()
{
    SwitchToThread();
}

//  a hint to the processor inside a spin loop
inline void
relax()
{
    YieldProcessor();
}

//  starts fn(arg) on a new thread (detached); false if it could not
inline bool
start(void (*fn)(void *), void *arg)
{
    struct Start
    {
	static DWORD WINAPI
	run(LPVOID p)
	{
	    void **a = static_cast<void **>(p);
	    void (*f)(void *) = reinterpret_cast<void (*)(void *)>(a[0]);
	    void *x = a[1];
	    std::free(a);
	    f(x);
	    return 0;
	}
    };
    void **a = static_cast<void **>(std::malloc(2 * sizeof(void *)));
    if (!a) {
	return false;
    }
    a[0] = reinterpret_cast<void *>(fn);
    a[1] = arg;
    HANDLE h = CreateThread(nullptr, 0, &Start::run, a, 0, nullptr);
    if (!h) {
	std::free(a);
	return false;
    }
    CloseHandle(h);
    return true;
}

#else

struct Mutex
{
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    void
    acquire()
    {
	pthread_mutex_lock(&mutex);
    }
    bool
    tryAcquire()
    {
	return pthread_mutex_trylock(&mutex) == 0;
    }
    void
    release()
    {
	pthread_mutex_unlock(&mutex);
    }
};

struct Condition
{
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

    void
    wait(Mutex &mutex)
    {
	pthread_cond_wait(&cond, &mutex.mutex);
    }
    void
    broadcast()
    {
	pthread_cond_broadcast(&cond);
    }
};

inline void
yield()
{
    sched_yield();
}

//  a hint to the processor inside a spin loop
inline void
relax()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

inline bool
start(void (*fn)(void *), void *arg)
{
    struct Start
    {
	static void *
	run(void *p)
	{
	    void **a = static_cast<void **>(p);
	    void (*f)(void *) = reinterpret_cast<void (*)(void *)>(a[0]);
	    void *x = a[1];
	    std::free(a);
	    f(x);
	    return nullptr;
	}
    };
    void **a = static_cast<void **>(std::malloc(2 * sizeof(void *)));
    if (!a) {
	return false;
    }
    a[0] = reinterpret_cast<void *>(fn);
    a[1] = arg;
    pthread_t thread;
    if (pthread_create(&thread, nullptr, &Start::run, a) != 0) {
	std::free(a);
	return false;
    }
    pthread_detach(thread);
    return true;
}

#endif

//  monotonic time in nanoseconds
inline std::int64_t
nanoseconds()
{
    return std::int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count());
}

//  ULMBLAS_SPIN_US: how long an idle worker spins before it sleeps, in
//  microseconds (default 100000, 0.1 s, as OpenBLAS's THREAD_TIMEOUT 28:
//  2^28 cycles); read once
inline std::int64_t
idleSpinNanoseconds()
{
    static const std::int64_t ns = [] {
	const char *value = std::getenv("ULMBLAS_SPIN_US");
	const std::int64_t us =
	    value ? std::int64_t(std::strtoll(value, nullptr, 10))
	          : std::int64_t(100000);
	return (us > 0 ? us : 0) * std::int64_t(1000);
    }();
    return ns;
}

//
//  Waiting for another thread inside a parallel region: spinning (relax),
//  after yieldAfter nanoseconds also yielding the processor (there may be
//  more threads than CPUs).  The clock is read every 256 iterations.
//
class Backoff
{
  public:
    explicit Backoff(std::int64_t yieldAfter = 1000000)
        : yieldAfter(yieldAfter), start(-1), count(0)
    {
    }

    void
    pause()
    {
	relax();
	if ((++count & 255) != 0) {
	    return;
	}
	const std::int64_t t = nanoseconds();
	if (start < 0) {
	    start = t;
	} else if (t - start > yieldAfter) {
	    yield();
	}
    }

    //  true after limit nanoseconds of pause (checked every 256 calls)
    bool
    expired(std::int64_t limit) const
    {
	return start >= 0 && (count & 255) == 0 &&
	       nanoseconds() - start > limit;
    }

  private:
    const std::int64_t yieldAfter;
    std::int64_t start;
    std::int64_t count;
};

//
//  Linux: the CPUs of the process's affinity mask, cores before their
//  second hardware threads (a CPU is the first of its core if it is the
//  first of its thread_siblings_list); determined once, when the pool is
//  created (the threads of the pool are pinned only during their parallel
//  regions, see ThreadPool).
//
struct CpuOrder
{
    std::ptrdiff_t count = 0;
#if defined(__linux__)
    std::ptrdiff_t cpu[CPU_SETSIZE];
    //  the first CPU of the core of each CPU (of the mask)
    std::ptrdiff_t core[CPU_SETSIZE];
    cpu_set_t all;
#endif

    static const CpuOrder &
    instance()
    {
	static const CpuOrder order = make();
	return order;
    }

  private:
    static CpuOrder
    make()
    {
	CpuOrder o;
#if defined(__linux__)
	if (sched_getaffinity(0, sizeof(o.all), &o.all) != 0) {
	    return o;
	}
	for (std::ptrdiff_t pass = 0; pass < 2; ++pass) {
	    for (std::ptrdiff_t c = 0; c < CPU_SETSIZE; ++c) {
		if (!CPU_ISSET(c, &o.all)) {
		    continue;
		}
		char path[128];
		std::snprintf(path, sizeof(path),
		              "/sys/devices/system/cpu/cpu%td/topology/"
		              "thread_siblings_list",
		              c);
		std::ptrdiff_t first = c;
		if (FILE *f = std::fopen(path, "r")) {
		    std::int64_t x;
		    if (std::fscanf(f, "%" SCNd64, &x) == 1) {
			first = std::ptrdiff_t(x);
		    }
		    std::fclose(f);
		}
		o.core[c] = first;
		if ((pass == 0) == (first == c)) {
		    o.cpu[o.count++] = c;
		}
	    }
	}
#endif
	return o;
    }
};

//
//  The affinity setting (ulmblas_set_affinity, ULMBLAS_AFFINITY): 0 none,
//  1 compact (thread tid on the tid-th CPU of CpuOrder: one thread per core
//  first), 2 (default) placed once: the workers run the first region after
//  the change pinned as with 1 and are unpinned after it (they keep their
//  CPUs as long as they keep running); other values: the default.  A change
//  increments the generation; the threads of the pool compare it at the start
//  of each parallel region. Effective on Linux only.
//
struct Affinity
{
    std::atomic<std::ptrdiff_t> mode;
    std::atomic<std::ptrdiff_t> generation;

    static Affinity &
    instance()
    {
	static Affinity affinity;
	return affinity;
    }

    void
    set(std::ptrdiff_t m)
    {
	mode.store(m == 0 || m == 1 ? m : 2, std::memory_order_relaxed);
	generation.fetch_add(1, std::memory_order_release);
    }

  private:
    Affinity() : mode(2), generation(0)
    {
	const char *a = std::getenv("ULMBLAS_AFFINITY");
	if (a && (a[0] == '0' || a[0] == '1') && a[1] == 0) {
	    mode.store(a[0] - '0', std::memory_order_relaxed);
	}
    }
};

//
//  Linux: true if the calling thread runs on the core of the CPU of one of
//  the threads 1, ..., nt-1 of a region (or on an unknown CPU); false
//  elsewhere.  A caller that is not pinned for a region (ThreadPool::run
//  with pinCaller = false) is pinned if it would share a core with a worker
//  (it then usually stays on its CPU afterwards: the next regions need no
//  system calls).
//
inline bool
callerSharesCore(std::ptrdiff_t nt)
{
#if defined(__linux__)
    const CpuOrder &order = CpuOrder::instance();
    if (order.count == 0) {
	return false;
    }
    const std::ptrdiff_t c = sched_getcpu();
    if (c < 0 || c >= CPU_SETSIZE || !CPU_ISSET(c, &order.all)) {
	return true;
    }
    for (std::ptrdiff_t t = 1; t < nt; ++t) {
	if (order.core[order.cpu[t % order.count]] == order.core[c]) {
	    return true;
	}
    }
#else
    (void)nt;
#endif
    return false;
}

//  the calling thread's CPU mask (Linux; nothing elsewhere)
struct CpuMask
{
#if defined(__linux__)
    cpu_set_t set;
#endif

    void
    save()
    {
#if defined(__linux__)
	sched_getaffinity(0, sizeof(set), &set);
#endif
    }

    void
    restore() const
    {
#if defined(__linux__)
	sched_setaffinity(0, sizeof(set), &set);
#endif
    }
};

//  Linux: pins the calling thread to the tid-th CPU of CpuOrder (pin), or
//  allows it all CPUs of the process (unpin); nothing elsewhere
inline void
pin(std::ptrdiff_t tid)
{
#if defined(__linux__)
    const CpuOrder &order = CpuOrder::instance();
    if (order.count == 0) {
	return;
    }
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(order.cpu[tid % order.count], &one);
    sched_setaffinity(0, sizeof(one), &one);
#else
    (void)tid;
#endif
}

inline void
unpin()
{
#if defined(__linux__)
    const CpuOrder &order = CpuOrder::instance();
    if (order.count > 0) {
	sched_setaffinity(0, sizeof(order.all), &order.all);
    }
#endif
}

} // namespace threads

//  the default number of threads (see above)
inline std::ptrdiff_t
defaultNumThreads()
{
    const char *names[] = {"ULMBLAS_NUM_THREADS", "OMP_NUM_THREADS"};
    for (const char *name : names) {
	if (const char *value = std::getenv(name)) {
	    const std::ptrdiff_t n =
	        std::ptrdiff_t(std::strtoll(value, nullptr, 10));
	    if (n > 0) {
		return n;
	    }
	}
    }
#if defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
	const std::ptrdiff_t cpus = CPU_COUNT(&set);
	if (cpus > 0) {
	    return cpus;
	}
    }
#elif defined(__APPLE__)
    std::int32_t cores = 0;
    std::size_t size = sizeof(cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, nullptr, 0) ==
            0 &&
        cores > 0) {
	return std::ptrdiff_t(cores);
    }
#elif defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    if (info.dwNumberOfProcessors > 0) {
	return std::ptrdiff_t(info.dwNumberOfProcessors);
    }
#endif
    return 1;
}

class Barrier
{
  public:
    explicit Barrier(std::ptrdiff_t numThreads)
        : n(numThreads), count(0), generation(0)
    {
    }

    void
    wait()
    {
	if (n <= 1) {
	    return;
	}
	const std::ptrdiff_t gen = generation.load(std::memory_order_acquire);
	if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
	    count.store(0, std::memory_order_relaxed);
	    generation.fetch_add(1, std::memory_order_release);
	    return;
	}
	threads::Backoff backoff;
	while (generation.load(std::memory_order_acquire) == gen) {
	    backoff.pause();
	}
    }

  private:
    const std::ptrdiff_t n;
    std::atomic<std::ptrdiff_t> count;
    std::atomic<std::ptrdiff_t> generation;
};

//
//  TreeBarrier: a barrier for many threads (thread tid of numThreads, at
//  most 1024): the arrivals are counted in groups of Fanin threads, the
//  last thread of a group arrives at the group of the next level (a
//  combining tree), the last one at the root releases all threads (one
//  generation counter).  A single counter (Barrier) is a cache line that
//  all threads modify one after the other: on an AMD EPYC 9454 (2 x 48
//  cores) a barrier of 96 threads took 11.8 us with Barrier, 2.5 us with
//  TreeBarrier (Intel Xeon 8358, 64 threads: 11 and 1.5 us).
//
//  Correctness: the last thread of a group resets its counter before it
//  arrives at the next level (an acquire-release operation), the root's
//  release follows all of these, and a thread arrives again only after it
//  has seen the release; a thread that is not the last of its group waits
//  for the release.  Trivially destructible.
//
class TreeBarrier
{
  public:
    explicit TreeBarrier(std::ptrdiff_t numThreads) : n(numThreads) {}

    void
    wait(std::ptrdiff_t tid)
    {
	if (n <= 1) {
	    return;
	}
	const std::ptrdiff_t gen =
	    generation.value.load(std::memory_order_acquire);
	//  the level: its first counter (base), its number of members
	//  (width), the member (id)
	std::ptrdiff_t base = 0, width = n, id = tid;
	for (;;) {
	    const std::ptrdiff_t group = id / Fanin;
	    const std::ptrdiff_t rest = width - group * Fanin;
	    const std::ptrdiff_t size = rest < Fanin ? rest : Fanin;
	    Counter &c = count[base + group];
	    if (c.value.fetch_add(1, std::memory_order_acq_rel) != size - 1) {
		break;
	    }
	    c.value.store(0, std::memory_order_relaxed);
	    const std::ptrdiff_t groups = (width + Fanin - 1) / Fanin;
	    if (groups == 1) {
		generation.value.fetch_add(1, std::memory_order_acq_rel);
		return;
	    }
	    base += groups;
	    width = groups;
	    id = group;
	}
	threads::Backoff backoff;
	while (generation.value.load(std::memory_order_acquire) == gen) {
	    backoff.pause();
	}
    }

  private:
    static constexpr std::ptrdiff_t Fanin = 8;
    //  1024 threads: 128 + 16 + 2 + 1 groups
    static constexpr std::ptrdiff_t MaxGroups = 160;

    //  a cache line of its own; padded explicitly, so that the compiler adds
    //  no padding (no warning, e.g. MSVC C4324)
    struct alignas(128) Counter
    {
	std::atomic<std::ptrdiff_t> value;
	char pad[128 - sizeof(std::atomic<std::ptrdiff_t>)];

	Counter() : value(0) {}
    };
    static_assert(sizeof(Counter) == 128, "Counter: one 128-byte block");

    const std::ptrdiff_t n;
    Counter count[MaxGroups];
    Counter generation;
};

class ThreadPool
{
  public:
    //  the pool of the process (never destroyed: its workers wait until
    //  the process ends); static storage, no allocation
    static ThreadPool &
    instance()
    {
	static ThreadPool pool(defaultNumThreads());
	return pool;
    }

    //  the maximal number of threads of a parallel region
    std::ptrdiff_t
    size() const
    {
	return numThreads.load(std::memory_order_relaxed);
    }

    //  n > 0: at most n threads (and MaxThreads); n <= 0: the default
    void
    setSize(std::ptrdiff_t n)
    {
	if (n <= 0) {
	    n = defaultThreads;
	}
	numThreads.store(n < MaxThreads ? n : MaxThreads,
	                 std::memory_order_relaxed);
    }

    //  f(tid, nt) on nt threads (at most size(); see above).  pinCaller =
    //  false (short regions, e.g. of level 2): the calling thread is pinned
    //  only if it runs on the core of another thread of the region
    //  (threads::callerSharesCore); saving, setting and restoring its CPU
    //  mask are three system calls, an empty region on 6 threads of heim
    //  took 1.75 instead of 0.37 us.  Not pinning it at all was worse: on
    //  the core of a worker (placed there before) dgemv n = 1000 on 6
    //  threads ran at 16-23 instead of 57 GFLOP/s.
    template <typename F>
    void
    run(std::ptrdiff_t nt, F &f, bool pinCaller = true)
    {
	if (nt > size()) {
	    nt = size();
	}
	if (nt <= 1 || insideRegion() || !busy.tryAcquire()) {
	    f(std::ptrdiff_t(0), std::ptrdiff_t(1));
	    return;
	}
	startWorkers(nt - 1);
	if (nt > workers + 1) {
	    nt = workers + 1;
	}
	threads::Affinity &affinity = threads::Affinity::instance();
	const std::ptrdiff_t affinityGeneration =
	    affinity.generation.load(std::memory_order_acquire);
	const bool pinned =
	    affinity.mode.load(std::memory_order_relaxed) != 0 &&
	    (pinCaller || threads::callerSharesCore(nt));
	threads::CpuMask callerMask;
	if (pinned) {
	    callerMask.save();
	    threads::pin(0);
	}

	//  the job as a sequence lock: generation odd while the job is
	//  written, even (the next) when it is complete (see work)
	mutex.acquire();
	const std::size_t g = jobGeneration.load(std::memory_order_relaxed);
	jobGeneration.store(g + 1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	job.store(&call<F>, std::memory_order_relaxed);
	jobData.store(&f, std::memory_order_relaxed);
	jobThreads.store(nt, std::memory_order_relaxed);
	jobAffinity.store(affinityGeneration, std::memory_order_relaxed);
	remaining.store(nt - 1, std::memory_order_relaxed);
	jobGeneration.store(g + 2, std::memory_order_release);
	mutex.release();
	wakeup.broadcast();

	insideRegion() = true;
	f(std::ptrdiff_t(0), nt);
	insideRegion() = false;
	threads::Backoff backoff;
	while (remaining.load(std::memory_order_acquire) > 0) {
	    backoff.pause();
	}
	if (pinned) {
	    callerMask.restore();
	}
	busy.release();
    }

  private:
    explicit ThreadPool(std::ptrdiff_t n)
        : defaultThreads(n < MaxThreads ? n : MaxThreads),
          numThreads(n < MaxThreads ? n : MaxThreads), workers(0), job(nullptr),
          jobData(nullptr), jobThreads(0), jobAffinity(0), jobGeneration(0),
          remaining(0)
    {
	//  the CPUs of the process, before any thread is pinned
	threads::CpuOrder::instance();
#if !defined(_WIN32)
	pthread_atfork(nullptr, nullptr, &afterForkInChild);
#endif
    }

    //  the child of fork() has only the thread that called it: no workers,
    //  locks of the parent's threads as if free (e.g. Python's
    //  multiprocessing; the workers are started again when needed)
    static void
    afterForkInChild()
    {
	ThreadPool &pool = instance();
	pool.workers = 0;
	new (&pool.busy) threads::Mutex();
	new (&pool.mutex) threads::Mutex();
	new (&pool.wakeup) threads::Condition();
	const std::size_t g =
	    pool.jobGeneration.load(std::memory_order_relaxed);
	pool.jobGeneration.store(g + (g & 1), std::memory_order_relaxed);
	pool.remaining.store(0, std::memory_order_relaxed);
	insideRegion() = false;
    }

    template <typename F>
    static void
    call(void *f, std::ptrdiff_t tid, std::ptrdiff_t nt)
    {
	(*static_cast<F *>(f))(tid, nt);
    }

    static bool &
    insideRegion()
    {
	static thread_local bool inside = false;
	return inside;
    }

    //  worker tid: its pool, its number and the job generation when it
    //  was started (jobs up to it are not its own)
    struct Worker
    {
	ThreadPool *pool;
	std::ptrdiff_t tid;
	std::size_t generation;
    };

    //  at least n workers (if they can be started); called by the thread
    //  that holds busy, before it publishes the next job
    void
    startWorkers(std::ptrdiff_t n)
    {
	//  the arguments live as long as the pool (static storage)
	static Worker args[MaxThreads];
	while (workers < n && workers + 1 < MaxThreads) {
	    const std::ptrdiff_t tid = workers + 1;
	    args[tid] = Worker{this, tid,
	                       jobGeneration.load(std::memory_order_relaxed)};
	    if (!threads::start(&work, &args[tid])) {
		break;
	    }
	    ++workers;
	}
    }

    static void
    work(void *arg)
    {
	Worker *w = static_cast<Worker *>(arg);
	ThreadPool &pool = *w->pool;
	threads::Affinity &affinity = threads::Affinity::instance();
	insideRegion() = true;
	//  a worker started after earlier jobs must not take the last one
	//  (finished, its data gone) for a new one
	std::size_t seen = w->generation;
	//  the affinity generation this thread follows (none yet)
	std::ptrdiff_t pinnedFor = -1;
	bool pinned = false;
	for (;;) {
	    //  the next job (a generation other than seen, even: complete):
	    //  spinning for a while (a thread that keeps running keeps its
	    //  CPU, as OpenBLAS's threads), then sleeping (the caller
	    //  writes the job while it holds the mutex)
	    std::size_t generation =
	        pool.jobGeneration.load(std::memory_order_acquire);
	    {
		//  no yielding: the thread keeps its CPU while it spins
		threads::Backoff backoff(INT64_MAX);
		const std::int64_t limit = threads::idleSpinNanoseconds();
		while ((generation == seen || (generation & 1)) &&
		       !backoff.expired(limit)) {
		    backoff.pause();
		    generation =
		        pool.jobGeneration.load(std::memory_order_acquire);
		}
	    }
	    if (generation == seen || (generation & 1)) {
		pool.mutex.acquire();
		while ((generation = pool.jobGeneration.load(
		            std::memory_order_acquire)) == seen) {
		    pool.wakeup.wait(pool.mutex);
		}
		pool.mutex.release();
	    }
	    //  the job of this generation; if the caller has begun a newer
	    //  one meanwhile, this thread was not part of this one (a job
	    //  ends only when its threads have run it): the next
	    void (*f)(void *, std::ptrdiff_t, std::ptrdiff_t) =
	        pool.job.load(std::memory_order_relaxed);
	    void *data = pool.jobData.load(std::memory_order_relaxed);
	    const std::ptrdiff_t nt =
	        pool.jobThreads.load(std::memory_order_relaxed);
	    const std::ptrdiff_t jobAffinity =
	        pool.jobAffinity.load(std::memory_order_relaxed);
	    std::atomic_thread_fence(std::memory_order_acquire);
	    if (pool.jobGeneration.load(std::memory_order_relaxed) !=
	        generation) {
		continue;
	    }
	    seen = generation;
	    if (w->tid < nt) {
		bool placeOnce = false;
		if (jobAffinity != pinnedFor) {
		    pinnedFor = jobAffinity;
		    const std::ptrdiff_t mode =
		        affinity.mode.load(std::memory_order_relaxed);
		    if (mode != 0) {
			threads::pin(w->tid);
		    } else if (pinned) {
			threads::unpin();
		    }
		    pinned = mode == 1;
		    placeOnce = mode == 2;
		}
		f(data, w->tid, nt);
		if (placeOnce) {
		    threads::unpin();
		}
		pool.remaining.fetch_sub(1, std::memory_order_acq_rel);
	    }
	}
    }

    static constexpr std::ptrdiff_t MaxThreads = 1024;

    const std::ptrdiff_t defaultThreads;
    std::atomic<std::ptrdiff_t> numThreads;
    std::ptrdiff_t workers;
    threads::Mutex busy;
    threads::Mutex mutex;
    threads::Condition wakeup;
    std::atomic<void (*)(void *, std::ptrdiff_t, std::ptrdiff_t)> job;
    std::atomic<void *> jobData;
    std::atomic<std::ptrdiff_t> jobThreads;
    std::atomic<std::ptrdiff_t> jobAffinity;
    std::atomic<std::size_t> jobGeneration;
    std::atomic<std::ptrdiff_t> remaining;
};

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_THREADPOOL_H
