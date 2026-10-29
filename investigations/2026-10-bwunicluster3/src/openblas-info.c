/*  openblas-info: core type and configuration that OpenBLAS uses here. */
#include <stdio.h>

char *openblas_get_corename(void);
char *openblas_get_config(void);
int   openblas_get_num_threads(void);

int
main(void)
{
    printf("corename %s\nconfig %s\nthreads %d\n", openblas_get_corename(),
           openblas_get_config(), openblas_get_num_threads());
    return 0;
}
