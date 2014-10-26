all: f77blas refblas cblas clapack
	$(MAKE) -C bench

.PHONY: f77blas refblas cblas clapack

refblas:
	$(MAKE) -C refblas

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
	$(MAKE) -C refblas clean
	$(MAKE) -C test/F77 clean
