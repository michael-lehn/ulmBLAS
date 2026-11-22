// Store buffering: thread 1 stores x[i] and then loads y[i], thread 2
// stores y[i] and then loads x[i].  In every interleaving of the four
// operations at least one load comes after the other thread's store, so
// r1 = r2 = 0 seems impossible.  But a store may wait in the store buffer
// of its core while the following load already takes its value from
// memory.  relaxed and release/acquire allow this result; only
// memory_order_seq_cst forbids it.
//
// Both threads run through N slots without waiting for each other; a hit
// is a slot i with r1[i] = r2[i] = 0.
#include <atomic>
#include <cstdio>
#include <memory>
#include <thread>

template <std::memory_order Store, std::memory_order Load>
long
hits(long N)
{
    std::unique_ptr<std::atomic<int>[]> x(new std::atomic<int>[N]);
    std::unique_ptr<std::atomic<int>[]> y(new std::atomic<int>[N]);
    std::unique_ptr<char[]> r1(new char[N]), r2(new char[N]);
    for (long i = 0; i < N; ++i) {
	x[i].store(0, std::memory_order_relaxed);
	y[i].store(0, std::memory_order_relaxed);
    }
    std::atomic<int> go(0);

    std::thread t1([&] {
	while (go.load() == 0) {
	}
	for (long i = 0; i < N; ++i) {
	    x[i].store(1, Store);
	    r1[i] = char(y[i].load(Load));
	}
    });
    std::thread t2([&] {
	while (go.load() == 0) {
	}
	for (long i = 0; i < N; ++i) {
	    y[i].store(1, Store);
	    r2[i] = char(x[i].load(Load));
	}
    });
    go.store(1);
    t1.join();
    t2.join();

    long count = 0;
    for (long i = 0; i < N; ++i) {
	count += r1[i] == 0 && r2[i] == 0;
    }
    return count;
}

int
main()
{
    const long N = 20000000;
    const std::memory_order relaxed = std::memory_order_relaxed;
    const std::memory_order release = std::memory_order_release;
    const std::memory_order acquire = std::memory_order_acquire;
    const std::memory_order seq_cst = std::memory_order_seq_cst;

    std::printf("r1 = r2 = 0 in %ld slots, three runs each\n", N);
    for (int run = 0; run < 3; ++run) {
	std::printf("relaxed: %6ld   release/acquire: %6ld   seq_cst: %6ld\n",
	            hits<relaxed, relaxed>(N), hits<release, acquire>(N),
	            hits<seq_cst, seq_cst>(N));
    }
}
