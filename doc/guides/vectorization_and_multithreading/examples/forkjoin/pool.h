#ifndef POOL_H
#define POOL_H 1

// A minimal thread pool with the interface of ulmBLAS's ThreadPool
// (ulmblas/auxiliary/threadpool.h):
//
//     pool.run(nt, f)   calls f(tid, nt) on nt threads, tid = 0, ..., nt-1,
//                       and returns when all calls have returned.  The
//                       calling thread is thread 0.  Inside a region (nested
//                       call) f(0, 1) runs on the calling thread.
//
// Simplifications: every worker acknowledges every job (also the workers
// with tid >= nt), so the next job is published only after all workers
// have read the current one; idle workers spin and yield, they never
// sleep.  ulmBLAS uses a sequence lock instead and lets idle workers sleep
// on a condition variable after 0.1 s.

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

class ThreadPool
{
  public:
    explicit ThreadPool(int maxThreads)
    {
	for (int tid = 1; tid < maxThreads; ++tid) {
	    workers.emplace_back([this, tid] { work(tid); });
	}
    }

    ~ThreadPool()
    {
	stop.store(true, std::memory_order_relaxed);
	generation.fetch_add(1, std::memory_order_release);
	for (std::thread &t : workers) {
	    t.join();
	}
    }

    int
    size() const
    {
	return int(workers.size()) + 1;
    }

    template <typename F>
    void
    run(int nt, F &f)
    {
	if (nt > size()) {
	    nt = size();
	}
	if (nt <= 1 || insideRegion()) {
	    f(0, 1);
	    return;
	}
	// publish the job: the plain stores before the release increment of
	// generation are visible to every worker that acquires the new value
	job = &call<F>;
	jobData = &f;
	jobThreads = nt;
	remaining.store(int(workers.size()), std::memory_order_relaxed);
	generation.fetch_add(1, std::memory_order_release);

	insideRegion() = true;
	f(0, nt);
	insideRegion() = false;

	// join: wait until every worker has acknowledged the job
	while (remaining.load(std::memory_order_acquire) > 0) {
	    std::this_thread::yield();
	}
    }

  private:
    template <typename F>
    static void
    call(void *f, int tid, int nt)
    {
	(*static_cast<F *>(f))(tid, nt);
    }

    static bool &
    insideRegion()
    {
	static thread_local bool inside = false;
	return inside;
    }

    void
    work(int tid)
    {
	insideRegion() = true;
	std::size_t seen = 0;
	for (;;) {
	    std::size_t g;
	    while ((g = generation.load(std::memory_order_acquire)) == seen) {
		std::this_thread::yield();
	    }
	    seen = g;
	    if (stop.load(std::memory_order_relaxed)) {
		return;
	    }
	    if (tid < jobThreads) {
		job(jobData, tid, jobThreads);
	    }
	    remaining.fetch_sub(1, std::memory_order_release);
	}
    }

    std::vector<std::thread> workers;
    // the job: written by run before generation is incremented
    void (*job)(void *, int, int) = nullptr;
    void *jobData = nullptr;
    int jobThreads = 0;

    std::atomic<std::size_t> generation{0};
    std::atomic<int> remaining{0};
    std::atomic<bool> stop{false};
};

#endif // POOL_H
