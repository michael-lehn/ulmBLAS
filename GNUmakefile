#
#  ulmBLAS -- requires GNU Make 3.81 or newer (see make.inc).
#
#  GNU Make reads this file.  Other implementations of make read Makefile,
#  which only explains that GNU Make is required.
#
#  Run 'make help' for a list of targets.
#
TOPDIR = .

include $(TOPDIR)/make.inc

.PHONY : all cblas atlblas clapack atllapack f77blas refblas \
         check check-numeric check-netlib check_cblas check_f77blas \
         bench benchmark-suite clean help

all : cblas clapack

include $(TOPDIR)/mk/tools.mk

#
#  Libraries
#
cblas : tool-ar
	$(MAKE) -C interfaces/blas/C config_cblas

atlblas : tool-ar
	$(MAKE) -C interfaces/blas/C config_atl

clapack : cblas
	$(MAKE) -C interfaces/lapack/C config_cblas

atllapack : atlblas
	$(MAKE) -C interfaces/lapack/C config_atl

f77blas : tool-ar tool-fc
	$(MAKE) -C interfaces/blas/F77

refblas : tool-ar tool-fc
	$(MAKE) -C refblas

#
#  Tests (see test/Makefile)
#
check check-numeric check-netlib :
	$(MAKE) -C test $@

check_cblas :
	$(MAKE) -C test check-netlib NETLIB_SUITES=C

check_f77blas :
	$(MAKE) -C test check-netlib NETLIB_SUITES=F77

#
#  Benchmarks (not part of 'make check').  All libraries the benchmark
#  variants need are built here, each by its own makefile, before the
#  benchmark programs are built.
#
BENCH_LIBS = cblas atlblas atllapack f77blas refblas

bench : $(BENCH_LIBS) tool-cxx-link tool-cc tool-fc-link
	$(MAKE) -C bench

benchmark-suite : $(BENCH_LIBS) tool-cxx-link tool-cc tool-fc-link
	$(MAKE) -C bench benchmark-suite

clean :
	$(MAKE) -C interfaces/blas/C clean
	$(MAKE) -C interfaces/blas/F77 clean
	$(MAKE) -C interfaces/lapack/C clean
	$(MAKE) -C refblas clean
	$(MAKE) -C test clean
	$(MAKE) -C bench clean

help :
	@echo 'Targets (GNU Make $(ULMBLAS_MAKE_MIN) or newer):'
	@echo '  all (default)    cblas and clapack'
	@echo '  cblas            C interface of BLAS:   libcblas.a, cblas.h'
	@echo '  clapack          C interface of LAPACK: libclapack.a, clapack.h'
	@echo '  atlblas          ATLAS compatible BLAS: libatlblas.a, atlblas.h'
	@echo '  atllapack        ATLAS compatible LAPACK: libatllapack.a, atllapack.h'
	@echo '  f77blas          Fortran 77 interface: libf77blas.a, f77blas.h (needs FC)'
	@echo '  refblas          Netlib reference BLAS: librefblas.a (needs FC)'
	@echo '  check            build and run all tests'
	@echo '  check-numeric    only the C++ tests (no Fortran compiler needed)'
	@echo '  check-netlib     only the Netlib BLAS test suites (needs FC and CC)'
	@echo '  bench            benchmarks (long running, see bench/)'
	@echo '  clean            remove all generated files'
	@echo
	@echo 'Variables: CXX CC FC AR ARFLAGS RANLIB CXXSTD ARCHFLAGS OPTFLAGS FOPTFLAGS'
	@echo '           COPTFLAGS CPPFLAGS CXXFLAGS CFLAGS FFLAGS LDFLAGS LDLIBS FLIBS'
	@echo '           (see make.inc)'
	@echo
	@echo 'Current settings:'
	@echo '  CXX       = $(CXX)'
	@echo '  CXXSTD    = $(CXXSTD)'
	@echo '  ARCHFLAGS = $(ARCHFLAGS)'
	@echo '  OPTFLAGS  = $(OPTFLAGS)'
	@echo '  CXXFLAGS  = $(CXXFLAGS)'
	@echo '  CC        = $(CC)'
	@echo '  FC        = $(FC)'
	@echo '  FOPTFLAGS = $(FOPTFLAGS)'
	@echo '  FFLAGS    = $(FFLAGS)'
