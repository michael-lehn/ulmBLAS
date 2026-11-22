/*  stubs for the ATLAS extension ger2 (not a BLAS function, not tested) */
#include <stdio.h>
#include <stdlib.h>
#define STUB(name)                                                             \
    void name(void)                                                            \
    {                                                                          \
	fprintf(stderr, #name ": not available\n");                            \
	abort();                                                               \
    }
STUB(ATL_cger2c)
STUB(ATL_cger2u)
STUB(ATL_zger2c)
STUB(ATL_zger2u)
