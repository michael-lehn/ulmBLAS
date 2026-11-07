/*
 * The ulmlapack task pattern against libomp bugs llvm#226798 and llvm#226812.
 *   parallel num_threads(th varying 2..8) + single (implicit barrier)
 *   dependent tasks with priorities, the last one with priority 3
 *   a "panel" task creating child tasks and waiting for them (taskwait)
 *   between regions: another parallel region with a different thread count
 *   (stands in for OpenBLAS's parallel regions)
 * argv[1] = mode: "master" (generator = primary, no barrier) or "single"
 * argv[2] = regions
 *
 * macOS, Homebrew libomp:
 *   cc -O1 test/libomp_pattern.c -Xpreprocessor -fopenmp \
 *      -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
 *   OMP_MAX_TASK_PRIORITY=3 ./a.out master 30000    # libomp 23.1.4: fails
 *   OMP_MAX_TASK_PRIORITY=3 ./a.out single 30000    # passes
 */
#include <omp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define N 512
static double a[N+2];
static long other;

static void spin(double s) { double t0 = omp_get_wtime(); while (omp_get_wtime()-t0 < s) {} }

static void body(void)
{
    double *x = a+N, *y = a+N+1;
    #pragma omp task depend(inout: x[0]) priority(3)
    {   /* panel-like: children, then taskwait */
        for (int j=0; j<4; ++j) {
            #pragma omp task priority(3)
            spin(2e-6);
        }
        #pragma omp taskwait
        spin(100e-6);
    }
    for (int i=0; i<N; ++i) {
        #pragma omp task depend(inout: a[i]) priority(i%3)
        a[i] += 1.0;
    }
    #pragma omp task depend(in: x[0]) depend(in: a[N-1]) priority(3)
    y[0] += 1.0;
}

static void on_alarm(int s) { (void)s; (void)!write(2, "HANG\n", 5); _exit(3); }

int main(int argc, char **argv)
{
    int single = strcmp(argv[1], "single")==0;
    long regions = atol(argv[2]);
    signal(SIGALRM, on_alarm);
    for (long r=0; r<regions; ++r) {
        alarm(10);
        int th = 2 + (int)(r % 7);
        if (single) {
            #pragma omp parallel num_threads(th)
            #pragma omp single
            body();
        } else {
            #pragma omp parallel num_threads(th)
            #pragma omp master
            body();
        }
        #pragma omp parallel num_threads(10 - th)
        {
            #pragma omp atomic
            other++;
        }
    }
    long bad = 0;
    for (int i=0; i<N; ++i) bad += a[i] != (double)regions;
    bad += a[N+1] != (double)regions;
    printf("%s %s: %ld regions, %ld wrong\n", bad ? "FAILED" : "PASSED", argv[1], regions, bad);
    return bad!=0;
}
