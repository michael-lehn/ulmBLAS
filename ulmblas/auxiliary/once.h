#ifndef ULMBLAS_AUXILIARY_ONCE_H
#define ULMBLAS_AUXILIARY_ONCE_H 1

//
//  Lazy<T>: an object of static storage that is constructed at the first
//  call of get (thread-safe) and never destroyed.
//
//  It replaces function-local statics with dynamic initialization
//  (static T x = f();): the compiler guards those with
//  __cxa_guard_acquire/__cxa_guard_release of the C++ runtime, and ulmBLAS
//  links no C++ runtime.  Lazy has a constexpr constructor and a trivial
//  destructor, so a static Lazy is initialized at compile time (no guard)
//  and not registered for destruction:
//
//      static Lazy<Platform> p;
//      return p.get([](void *at) { new (at) Platform(detectPlatform()); });
//
//  construct(at) constructs the object at the address at (placement new),
//  so T need not be copyable or movable (atomics, the thread pool).  The
//  first thread constructs it; threads that call get meanwhile wait.  As
//  with a function-local static, construct must not call get of the same
//  Lazy.
//

#include <atomic>

#if !defined(_WIN32)
#include <sched.h>
#endif

namespace ulmBLAS {

template <typename T> class Lazy
{
  public:
    constexpr Lazy() : state(idle), object(nullptr), storage{} {}

    template <typename Construct>
    T &
    get(Construct construct)
    {
	if (state.load(std::memory_order_acquire) != done) {
	    init(construct);
	}
	return *object;
    }

  private:
    enum
    {
	idle,
	busy,
	done
    };

    template <typename Construct>
    void
    init(Construct &construct)
    {
	int expected = idle;
	if (state.compare_exchange_strong(expected, busy,
	                                  std::memory_order_acquire)) {
	    construct(static_cast<void *>(storage));
	    object = reinterpret_cast<T *>(storage);
	    state.store(done, std::memory_order_release);
	    return;
	}
	while (state.load(std::memory_order_acquire) != done) {
#if defined(_WIN32)
#if defined(__x86_64__) || defined(__i386__)
	    __builtin_ia32_pause();
#endif
#else
	    sched_yield();
#endif
	}
    }

    std::atomic<int> state;
    T *object;
    alignas(T) unsigned char storage[sizeof(T)];
};

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_ONCE_H
