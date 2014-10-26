all: f77blas cblas clapack
	$(MAKE) -C bench

f77blas:
	$(MAKE) -C interfaces/blas/F77

cblas:
	$(MAKE) -C interfaces/blas/C

clapack:
	$(MAKE) -C interfaces/lapack/C

check: f77blas
	$(MAKE) -C test/F77
	$(MAKE) -C test/F77 check

bench: all
	$(MAKE) -C bench

clean:
	$(MAKE) -C interfaces/blas/C clean
	$(MAKE) -C interfaces/blas/F77 clean
	$(MAKE) -C interfaces/lapack/C clean
	$(MAKE) -C bench clean
	$(MAKE) -C test/F77 clean
