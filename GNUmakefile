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

.PHONY : tune tune-params all cblas atlblas clapack atllapack f77blas f77lapack refblas lbt shared \
         install-shared \
         check-shared check-dispatch check-large shared-dispatch \
         check check-numeric check-netlib check-reference-lapack \
         check-reference-lapack-lin \
         check-conformance check-protocol \
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

f77lapack : f77blas
	$(MAKE) -C interfaces/lapack/F77

refblas : tool-ar tool-fc
	$(MAKE) -C refblas

lbt : tool-ar
	$(MAKE) -C interfaces/blas/lbt

shared : tool-ar
	$(MAKE) -C interfaces/blas/shared

#  Apple Silicon and x86-64: libulmblas selects a backend (libulmblas_sme,
#  _neon; _avx512, _avx2, _sse3, _generic) at load time
shared-dispatch : tool-ar
	$(MAKE) -C interfaces/blas/dispatch

#  Build and install the shared libraries (libulmblas, libulmlapack if a
#  Fortran compiler is found), cblas.h and pkg-config files in PREFIX, e.g.
#  for NumPy (interfaces/python/numpy.sh)
install-shared :
	@test -n "$(PREFIX)" || { echo "usage: make install-shared PREFIX=dir"; exit 2; }
	MAKE="$(MAKE)" FC="$(if $(filter default,$(origin FC)),gfortran,$(FC))" \
	    $(SHELL) interfaces/install-shared.sh "$(PREFIX)"

#  The Netlib test programs (test/F77, test/C) against libulmblas (needs FC),
#  built in its own directory: in the top directory libulmblas.dylib (.so)
#  may be the link of make shared-dispatch to the dispatching library
check-shared : tool-ar
	$(MAKE) -C interfaces/blas/shared OUTDIR=$(abspath test/build/shared-lib) \
	        BUILD_DIR=$(abspath test/build/shared-lib/build)
	FC="$(FC)" CC="$(CC)" $(SHELL) test/check-shared.sh test/build/shared-lib \
	         test/build/shared-lib/build test/build/shared
	$(if $(filter 0,$(ILP64)),,FC="$(FC)" CC="$(CC)" NM="$(NM)" \
	    $(SHELL) test/check-ilp64.sh test/build/shared-lib test/build/ilp64)
	$(MAKE) -C interfaces/blas/shared INTERFACE64=1 \
	        OUTDIR=$(abspath test/build/shared-lib) \
	        BUILD_DIR=$(abspath test/build/shared-lib/build64)
	FC="$(FC)" CC="$(CC)" NM="$(NM)" $(SHELL) test/check-ilp64.sh \
	    test/build/shared-lib test/build/plain64 ulmblas64 - \
	    test/build/shared-lib/build64

#  The same test programs against the dispatching library (make
#  shared-dispatch, built in its own directory), without the CBLAS programs
#  of levels 2 and 3 (see test/check-shared.sh), and c_scal with each
#  backend the CPU supports
check-dispatch : tool-ar
	$(MAKE) -C interfaces/blas/dispatch OUTDIR=$(abspath test/build/dispatch-lib) \
	        BUILD_DIR=$(abspath test/build/dispatch-lib/build)
	inc=`ls test/build/dispatch-lib/build/*/cblas.h | head -1` && \
	FC="$(FC)" CC="$(CC)" $(SHELL) test/check-shared.sh test/build/dispatch-lib \
	         `dirname $$inc` test/build/dispatch ulmblas f77
	if [ -f test/build/dispatch-lib/build/interface64/include/cblas.h ]; then \
	    FC="$(FC)" CC="$(CC)" NM="$(NM)" $(SHELL) test/check-ilp64.sh \
	        test/build/dispatch-lib test/build/dispatch64 ulmblas64 front \
	        test/build/dispatch-lib/build/interface64/include; \
	fi

#  The full check with the ATLAS testers (bench/check; long running, see
#  bench/Makefile): every routine of levels 1 to 3, types s d c z, the CBLAS
#  and the Fortran 77 interface, against the reference BLAS
check-large : tool-ar
	$(MAKE) cblas f77blas refblas
	$(MAKE) -C bench check-large

#
#  Tests (see test/Makefile)
#
check check-numeric check-netlib check-reference-lapack check-conformance \
check-reference-lapack-lin :
	$(MAKE) -C test $@

#  The synchronization of the parallel gemm frame, all interleavings of
#  small cases (test/protocol/frame.py; FULL=1: larger cases, about a minute)
check-protocol :
	python3 test/protocol/frame.py $(if $(filter 1,$(FULL)),--full)

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
#  Tuning (tune/README.md): measures the GEMM candidates of this build
#  (ARCHFLAGS) on this machine and generates the selection table of its
#  platform (ulmblas/level3/gemm/tables/) and a report (tune/results/).
#  Takes several minutes; the machine should be idle.  Settings: PASSES,
#  ROUNDS, CHUNK, LOADMAX, PAUSE, PLATFORM, GRID.
#
tune : tool-cxx-link
	@CXX='$(CXX)' ARCHFLAGS='$(ARCHFLAGS)' OPTFLAGS='$(OPTFLAGS)' \
	 PASSES='$(PASSES)' ROUNDS='$(ROUNDS)' CHUNK='$(CHUNK)' \
	 LOADMAX='$(LOADMAX)' PAUSE='$(PAUSE)' PLATFORM='$(PLATFORM)' \
	 GRID='$(GRID)' \
	 $(SHELL) $(TOPDIR)/tune/tune.sh

#
#  Tuning of the other parameters (tune/README.md): threads, variants and
#  block sizes of ulmblas/config/tuned.h, measured on this machine group
#  by group; generates ulmblas/config/tuned/PLATFORM.h and a report
#  (tune/results/PLATFORM/params.md).  Settings: TUNEGROUPS, TYPES, PASSES,
#  ROUNDS, CHUNK, LOADMAX, PAUSE, PLATFORM, VERIFY; OPENMP=1 also measures
#  the tasks of potrf.
#
tune-params : tool-cxx-link
	@CXX='$(CXX)' ARCHFLAGS='$(ARCHFLAGS)' OPTFLAGS='$(OPTFLAGS)' \
	 SIMDFLAGS='$(SIMDFLAGS)' OMPFLAGS='$(OMPFLAGS)' OMPLIBS='$(OMPLIBS)' \
	 TUNEGROUPS='$(TUNEGROUPS)' TYPES='$(TYPES)' \
	 PASSES='$(PASSES)' ROUNDS='$(ROUNDS)' CHUNK='$(CHUNK)' \
	 LOADMAX='$(LOADMAX)' PAUSE='$(PAUSE)' PLATFORM='$(PLATFORM)' \
	 VERIFY='$(VERIFY)' \
	 $(SHELL) $(TOPDIR)/tune/params.sh

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
	$(MAKE) -C interfaces/lapack/F77 clean
	$(if $(wildcard refblas/Makefile),$(MAKE) -C refblas clean)
	$(MAKE) -C interfaces/blas/lbt clean
	$(MAKE) -C interfaces/blas/shared clean
	$(MAKE) -C interfaces/blas/dispatch clean
	$(MAKE) -C test clean
	$(if $(wildcard bench/Makefile),$(MAKE) -C bench clean)

help :
	@echo 'Targets (GNU Make $(ULMBLAS_MAKE_MIN) or newer):'
	@echo '  all (default)    cblas and clapack'
	@echo '  cblas            C interface of BLAS:   libcblas.a, cblas.h'
	@echo '  clapack          C interface of LAPACK: libclapack.a, clapack.h'
	@echo '  atlblas          ATLAS compatible BLAS: libatlblas.a, atlblas.h'
	@echo '  atllapack        ATLAS compatible LAPACK: libatllapack.a, atllapack.h'
	@echo '  f77blas          Fortran 77 interface: libf77blas.a, f77blas.h (needs FC)'
	@echo '  f77lapack        Fortran 77 interface of ?getrf, ?potrf: libf77lapack.a,'
	@echo '                   f77lapack.h (with libf77blas.a; needs FC)'
	@echo '  refblas          Netlib reference BLAS: librefblas.a (needs FC)'
	@echo '  lbt              shared library for Julia (libblastrampoline): ILP64'
	@echo '                   Fortran 77 interface, libulmblas_lbt.so / .dylib'
	@echo '  shared           shared library with the standard interfaces: Fortran 77'
	@echo '                   (LP64, dgemm_) and CBLAS, libulmblas.so / .dylib; also'
	@echo '                   ILP64 (dgemm_64_, cblas_dgemm_64) unless ILP64=0'
	@echo '  shared-dispatch  Apple Silicon and x86-64: libulmblas selects a backend'
	@echo '                   (sme, neon; avx512, avx2, sse3, generic) at load time'
	@echo '  check-shared     Netlib test programs against libulmblas (needs FC)'
	@echo '  check-dispatch   the same against the dispatching libulmblas, ?scal with'
	@echo '                   every backend the CPU supports (needs FC)'
	@echo '  check-large      the full check with the ATLAS testers (bench/check,'
	@echo '                   hours; needs FC and python3)'
	@echo '  check            build and run all tests'
	@echo '  check-numeric    only the C++ tests (no Fortran compiler needed)'
	@echo '  check-netlib     only the Netlib BLAS test suites (needs FC and CC)'
	@echo '  check-reference-lapack'
	@echo '                   test programs of Reference-LAPACK 3.12.1 (downloaded)'
	@echo '  check-reference-lapack-lin'
	@echo '                   ?getrf, ?potrf with the LIN test programs of'
	@echo '                   Reference-LAPACK 3.12.1 (paths GE, PO; downloaded)'
	@echo '  check-conformance'
	@echo '                   all BLAS routines compared with the reference BLAS'
	@echo '                   and CBLAS of LAPACK 3.12.1 (downloaded)'
	@echo '  check-protocol   synchronization of the parallel gemm, all interleavings'
	@echo '                   of small cases (python3; FULL=1: larger cases)'
	@echo '  check_arch       ARCHFLAGS this machine and CXX support, with a suggestion'
	@echo '  bench-gemm       GEMM benchmark of ulmBLAS and the BLAS libraries found'
	@echo '  tune             measure GEMM on this machine, generate its selection table'
	@echo '  tune-params      measure threads, variants, block sizes (ulmblas/config/tuned.h)'
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
	@echo '           OPENMP=1 (task-based ?getrf, ?potrf) OMPFLAGS OMPLIBS'
	@echo '           (see make.inc)'
	@echo
	@echo 'Current settings:'
	@echo '  CXX       = $(CXX)'
	@echo '  CXXSTD    = $(CXXSTD)'
	@echo '  ARCHFLAGS = $(ARCHFLAGS)'
	@echo '  OPENMP    = $(OPENMP)'
	@echo '  OPTFLAGS  = $(OPTFLAGS)'
	@echo '  CXXFLAGS  = $(CXXFLAGS)'
	@echo '  CC        = $(CC)'
	@echo '  FC        = $(FC)'
	@echo '  FOPTFLAGS = $(FOPTFLAGS)'
	@echo '  FFLAGS    = $(FFLAGS)'
