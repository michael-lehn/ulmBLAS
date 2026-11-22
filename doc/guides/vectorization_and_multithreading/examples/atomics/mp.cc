// Message passing: a writer stores data, then a flag; a reader waits for
// the flag, then loads the data.  With memory_order_relaxed the reader
// may see the flag but old data; with release/acquire (or fences) it
// cannot.
//
// The writer runs through N slots; the reader follows it.  A violation
// is a slot whose flag the reader saw set while its data was still 0.
#include <atomic>
#include <cstdio>
#include <memory>
#include <thread>

// In correct code only the flag must be atomic.  Here data is atomic as
// well because the relaxed mode would otherwise be a data race, that is
// undefined behavior.  Relaxed loads and stores compile to plain loads
// and stores, so the hardware sees the same program either way.
struct Slot
{
    std::atomic<int> data;
    std::atomic<int> flag;
};

enum class Mode
{
    Relaxed,
    ReleaseAcquire,
    Fences
};

long
violations(Mode mode, long N)
{
    std::unique_ptr<Slot[]> slot(new Slot[N]);
    for (long i = 0; i < N; ++i) {
	slot[i].data.store(0, std::memory_order_relaxed);
	slot[i].flag.store(0, std::memory_order_relaxed);
    }
    std::atomic<int> go(0);
    long bad = 0;

    std::thread writer([&] {
	while (go.load() == 0) {
	}
	for (long i = 0; i < N; ++i) {
	    slot[i].data.store(1, std::memory_order_relaxed);
	    if (mode == Mode::ReleaseAcquire) {
		slot[i].flag.store(1, std::memory_order_release);
	    } else if (mode == Mode::Fences) {
		std::atomic_thread_fence(std::memory_order_release);
		slot[i].flag.store(1, std::memory_order_relaxed);
	    } else {
		slot[i].flag.store(1, std::memory_order_relaxed);
	    }
	}
    });
    std::thread reader([&] {
	while (go.load() == 0) {
	}
	for (long i = 0; i < N; ++i) {
	    if (mode == Mode::ReleaseAcquire) {
		while (slot[i].flag.load(std::memory_order_acquire) == 0) {
		}
	    } else if (mode == Mode::Fences) {
		while (slot[i].flag.load(std::memory_order_relaxed) == 0) {
		}
		std::atomic_thread_fence(std::memory_order_acquire);
	    } else {
		while (slot[i].flag.load(std::memory_order_relaxed) == 0) {
		}
	    }
	    bad += slot[i].data.load(std::memory_order_relaxed) == 0;
	}
    });
    go.store(1);
    writer.join();
    reader.join();
    return bad;
}

int
main()
{
    const long N = 20000000;
    std::printf("relaxed:         %ld violations in %ld slots\n",
                violations(Mode::Relaxed, N), N);
    std::printf("release/acquire: %ld violations in %ld slots\n",
                violations(Mode::ReleaseAcquire, N), N);
    std::printf("fences:          %ld violations in %ld slots\n",
                violations(Mode::Fences, N), N);
}
