#ifndef ULMBLAS_LAPACK_TASKS_SYNC_H
#define ULMBLAS_LAPACK_TASKS_SYNC_H 1

//
//  Synchronization helpers of the task-based factorizations (OpenMP).
//

#include <atomic>
#include <chrono>

namespace ulmBLAS { namespace tasks {

//
//  Fences at the start and end of every task body (workaround for LLVM's
//  OpenMP runtime, libomp, at least up to 23.1.4).
//
//  When a task with dependencies is created, libomp checks each
//  predecessor's dependency node without a lock: if node->dn.task is
//  already NULL, the predecessor counts as finished and no edge is added
//  (__kmp_process_dep_list, __kmp_depnode_link_successor in
//  kmp_taskdeps.cpp).  The finishing thread stores that NULL in
//  __kmp_release_deps with a plain store, after the task body, with no
//  release operation in between (taking the ticket lock is only an
//  acquire).  On a weakly ordered CPU (AArch64) the NULL can become visible
//  before the stores of the task body, so the new task may start and read
//  stale data.  Seen on an Apple M5: Cholesky TRSMs that used a diagonal
//  element from before its square root, about 1 in 400 calls with 8 x 8
//  tiles; 0 in 19,333 calls with the fences.  x86 orders stores (TSO), and
//  there these fences generate no instructions.
//
//  The release fence orders the stores of the task body before every later
//  store of the finishing thread (including the runtime's NULL); the
//  acquire fence orders the loads of the task body after everything the
//  thread observed before it.
//
inline void
taskEnter()
{
    std::atomic_thread_fence(std::memory_order_acquire);
}

inline void
taskLeave()
{
    std::atomic_thread_fence(std::memory_order_release);
}

inline void
cpuRelax()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

inline double
seconds()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

//
//  Sense-reversing spin barrier for the threads of an LU panel.  The
//  threads are bound to cores and the barrier is only used inside a panel
//  whose team has been formed (see getrftasks.tcc), so spinning is cheap
//  and cannot wait for a thread that never comes.
//
class SpinBarrier
{
    public:
        explicit SpinBarrier(int n = 1) : count_(0), sense_(0), n_(n) {}

        void
        reset(int n)
        {
            n_ = n;
            count_.store(0);
            sense_.store(0);
        }

        void
        wait(int &localSense)
        {
            localSense = 1-localSense;
            if (count_.fetch_add(1, std::memory_order_acq_rel)==n_-1) {
                count_.store(0, std::memory_order_relaxed);
                sense_.store(localSense, std::memory_order_release);
            } else {
                while (sense_.load(std::memory_order_acquire)!=localSense) {
                    cpuRelax();
                }
            }
        }

    private:
        alignas(64) std::atomic<int>  count_;
        alignas(64) std::atomic<int>  sense_;
        int                           n_;
};

} } // namespace tasks, ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_SYNC_H
