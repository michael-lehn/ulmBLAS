/* Probe: C compiler. */
#include <stdlib.h>

int
main(void)
{
    double *x = (double *)malloc(2*sizeof(double));
    int    ok;

    x[0] = 1;
    x[1] = 2;
    ok = (x[0] + x[1] == 3);
    free(x);
    return ok ? 0 : 1;
}
