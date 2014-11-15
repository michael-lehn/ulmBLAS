all: f77blas cblas clapack

.PHONY: refblas f77blas cblas clapack

refblas:
	$(MAKE) -C refblas

f77blas:
	$(MAKE) -C interfaces/blas/F77

cblas:
	$(MAKE) -C interfaces/blas/C cblas

atlblas:
	$(MAKE) -C interfaces/blas/C atl

clapack:
	$(MAKE) -C interfaces/lapack/C cblas

atllapack:
	$(MAKE) -C interfaces/lapack/C atl

check:
	$(MAKE) -C test/F77

bench: atlblas atllapack
	$(MAKE) -C bench

clean:
	$(MAKE) -C interfaces/blas/C clean
	$(MAKE) -C interfaces/blas/F77 clean
	$(MAKE) -C interfaces/lapack/C clean
	$(MAKE) -C bench clean
	$(MAKE) -C refblas clean
	$(MAKE) -C test/F77 clean
