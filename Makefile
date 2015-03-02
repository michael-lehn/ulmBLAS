all: cblas clapack

.PHONY: refblas f77blas cblas clapack

refblas:
	$(MAKE) -C refblas

f77blas:
	$(MAKE) -C interfaces/blas/F77

cblas:
	$(MAKE) -C interfaces/blas/C config_cblas

atlblas:
	$(MAKE) -C interfaces/blas/C config_atl

clapack:
	$(MAKE) -C interfaces/lapack/C config_cblas

atllapack:
	$(MAKE) -C interfaces/lapack/C config_atl

check_f77blas:
	$(MAKE) -C test/F77

check_cblas:
	$(MAKE) -C test/C

check: check_f77blas check_cblas

bench: atlblas atllapack
	$(MAKE) -C bench

benchmark-suite: atlblas atllapack
	$(MAKE) -C bench benchmark-suite

clean:
	$(MAKE) -C interfaces/blas/C clean
	$(MAKE) -C interfaces/blas/F77 clean
	$(MAKE) -C interfaces/lapack/C clean
	$(MAKE) -C bench clean
	$(MAKE) -C refblas clean
	$(MAKE) -C test/F77 clean
	$(MAKE) -C test/C clean
