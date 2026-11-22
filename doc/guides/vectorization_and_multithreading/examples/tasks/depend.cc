// The depend clause: the order of tasks is derived from the variables
// they read (in) and write (out, inout), in the order of creation.
#include <atomic>
#include <cstdio>
#include <omp.h>
#include <thread>

std::atomic<int> position(0);
char order[32];

// a task records its name when it starts and ends
void
work(char name, int milliseconds)
{
    order[position++] = name;
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    order[position++] = name;
}

int
main()
{
    int a = 0, b = 0, c = 0;

#pragma omp parallel num_threads(4)
#pragma omp single
    {
#pragma omp task depend(out : a) // writes a
	work('A', 20);
#pragma omp task depend(in : a) depend(out : b) // reads a, writes b
	work('B', 30);
#pragma omp task depend(in : a) depend(out : c) // reads a, writes c
	work('C', 10);
#pragma omp task depend(in : b, c) // reads b and c
	work('D', 10);
#pragma omp task depend(inout : a) // after B and C have read a
	work('E', 10);
    }
    order[position] = 0;
    std::printf("A start, A end, ...: %s\n", order);

    // iterator (OpenMP 5.0): one task depends on the column of tiles
    // t[0], ..., t[3]
    double t[4] = {};
    position = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
    {
	for (int i = 0; i < 4; ++i) {
#pragma omp task depend(inout : t[i])
	    work(char('0' + i), 10 * (4 - i));
	}
#pragma omp task depend(iterator(ii = 0 : 4), in : t[ii])
	work('X', 10);
    }
    order[position] = 0;
    std::printf("tiles, then X:       %s\n", order);
    (void)t;
    (void)a;
    (void)b;
    (void)c;
}
