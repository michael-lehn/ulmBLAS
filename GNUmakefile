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

.PHONY : all cblas atlblas clapack atllapack f77blas refblas lbt \
         check check-numeric check-netlib check-reference-lapack \
         check_cblas check_f77blas \
         bench bench-gemm benchmark-suite add_benchmark_result \
         check_benchmark_results check_arch clean help

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

lbt : tool-ar
	$(MAKE) -C interfaces/blas/lbt

#
#  Tests (see test/Makefile)
#
check check-numeric check-netlib check-reference-lapack :
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

#
#  ARCHFLAGS configurations this machine and CXX support, the micro kernels
#  they select and a suggested one (mk/check_arch.sh).  Builds nothing.
#
check_arch :
	@CXX='$(CXX)' ARCHFLAGS='$(ARCHFLAGS)' ARCHFLAGS_ORIGIN='$(origin ARCHFLAGS)' \
	 ARCHFLAGS_DEFAULT='$(ULMBLAS_ARCHFLAGS_DEFAULT)' \
	 $(SHELL) $(TOPDIR)/mk/check_arch.sh

#
#  GEMM benchmark of ulmBLAS against the BLAS libraries found on this
#  machine (bench/gemm/README.md): builds libcblas.a with ARCHFLAGS (which
#  select the micro kernels), then runs bench/gemm/run.sh.  Settings: PREC,
#  LIBS, SIZES, SAMPLES, ROUNDS, CPU, OUT and the library directories, as
#  make variables or in the environment.
#
bench-gemm : cblas tool-cxx-link tool-cc
	@PREC='$(PREC)' LIBS='$(LIBS)' SIZES='$(SIZES)' SAMPLES='$(SAMPLES)' \
	 ROUNDS='$(ROUNDS)' CPU='$(CPU)' OUT='$(OUT)' CC='$(CC)' CXX='$(CXX)' \
	 ARCHFLAGS='$(ARCHFLAGS)' $(SHELL) $(TOPDIR)/bench/gemm/run.sh

#
#  Benchmark archive (benchmark_results/, see the README.md there): adds a
#  completed benchmark campaign, regenerates its plots (Gnuplot) and the
#  index.  Independent of the build and of 'make check'; benchmarks are not
#  run.
#
add_benchmark_result :
	@$(SHELL) $(TOPDIR)/mk/benchmark_result.sh $(TOPDIR)/benchmark_results \
	          '$(BENCHMARK_DIR)' '$(BENCHMARK_NAME)'

check_benchmark_results :
	@python3 $(TOPDIR)/mk/check_benchmark_results.py $(TOPDIR)

clean :
	$(MAKE) -C interfaces/blas/C clean
	$(MAKE) -C interfaces/blas/F77 clean
	$(MAKE) -C interfaces/lapack/C clean
	$(MAKE) -C refblas clean
	$(MAKE) -C interfaces/blas/lbt clean
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
	@echo '  lbt              shared library for Julia (libblastrampoline): ILP64'
	@echo '                   Fortran 77 interface, libulmblas_lbt.so / .dylib'
	@echo '  check            build and run all tests'
	@echo '  check-numeric    only the C++ tests (no Fortran compiler needed)'
	@echo '  check-netlib     only the Netlib BLAS test suites (needs FC and CC)'
	@echo '  check-reference-lapack'
	@echo '                   test programs of Reference-LAPACK 3.12.1 (downloaded)'
	@echo '  check_arch       ARCHFLAGS this machine and CXX support, with a suggestion'
	@echo '  bench-gemm       GEMM benchmark of ulmBLAS and the BLAS libraries found'
	@echo '                   (PREC=s|d|c|z LIBS=auto|"ulmblas openblas ..." SIZES=...,'
	@echo '                   see bench/gemm/README.md)'
	@echo '  bench            ATLAS timing programs (long running, see bench/)'
	@echo '  add_benchmark_result BENCHMARK_DIR=dir [BENCHMARK_NAME=name]'
	@echo '                   add a benchmark campaign to benchmark_results/'
	@echo '  check_benchmark_results'
	@echo '                   check names, numbers and references of benchmark_results/'
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
