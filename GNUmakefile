#
#  ulmBLAS -- requires GNU (GNU's Not Unix) Make 3.81 or newer (see
#  make.inc).
#
#  GNU Make reads this file.  Other implementations of make read Makefile,
#  which only explains that GNU Make is required.
#
#  The whole build is this one make: the rules of the libraries are
#  generated for each build (mk/lib.mk), no make calls another (see
#  mk/README.md).  Run 'make help' for a list of targets.
#
TOPDIR = .

include $(TOPDIR)/make.inc

.DEFAULT_GOAL := all

.PHONY : tune tune-params all cblas atlblas clapack atllapack f77blas \
         f77lapack refblas lbt shared shared64 shared-dispatch sonames \
         build-shared install-shared check-package \
         check-shared check-dispatch check-large \
         check check-numeric check-netlib check-reference-lapack \
         check-reference-lapack-lin \
         check-conformance check-protocol \
         check_cblas check_f77blas \
         bench bench-gemm benchmark-suite add_benchmark_result \
         check_benchmark_results check_arch clean help

include $(TOPDIR)/mk/tools.mk
include $(TOPDIR)/mk/lib.mk
include $(TOPDIR)/mk/shared.mk
include $(TOPDIR)/mk/dispatch.mk
include $(TOPDIR)/mk/lbt.mk

#
#  Static libraries in the top directory: the C interfaces of BLAS (Basic
#  Linear Algebra Subprograms) and of the LAPACK (Linear Algebra PACKage)
#  functions (cblas, clapack), their ATLAS (Automatically Tuned Linear
#  Algebra Software) compatible variants
#  (atlblas, atllapack, for the benchmarks), the Fortran 77 interfaces
#  (f77blas, f77lapack) and the Netlib reference BLAS (refblas, for the
#  benchmarks and check-large)
#
top.cblas.out   := .
top.cblas.obj   := interfaces/blas/C/build.cblas
top.cblas.asm   := yes
top.cblas.tools := tool-ar
$(eval $(call cblas_lib,top.cblas,cblas))

top.atl.out   := .
top.atl.obj   := interfaces/blas/C/build.atl
top.atl.asm   := yes
top.atl.tools := tool-ar
$(eval $(call cblas_lib,top.atl,atl))

top.clapack.out   := .
top.clapack.obj   := interfaces/lapack/C/build.cblas
top.clapack.asm   := yes
top.clapack.blas  := top.cblas
top.clapack.tools := tool-ar
$(eval $(call clapack_lib,top.clapack,cblas))

top.atllapack.out   := .
top.atllapack.obj   := interfaces/lapack/C/build.atl
top.atllapack.asm   := yes
top.atllapack.blas  := top.atl
top.atllapack.tools := tool-ar
$(eval $(call clapack_lib,top.atllapack,atl))

top.f77.out   := .
top.f77.obj   := interfaces/blas/F77/build.f77
top.f77.asm   := yes
top.f77.tools := tool-ar tool-fc
$(eval $(call f77blas_lib,top.f77,f77))

top.f77lapack.out   := .
top.f77lapack.obj   := interfaces/lapack/F77/build.f77
top.f77lapack.blas  := top.f77
top.f77lapack.tools := tool-ar tool-fc
$(eval $(call f77lapack_lib,top.f77lapack,f77))

#  Netlib reference BLAS (Fortran 77)
refblas.obj     := refblas/build
refblas.lib     := librefblas.a
refblas.objects := $(patsubst refblas/%.f,$(refblas.obj)/%.o,\
                              $(wildcard refblas/*.f))
lib.clean       += $(refblas.obj) $(refblas.lib)

$(refblas.lib) : $(refblas.objects)
	$(call make_archive,$@,$(refblas.objects))

$(refblas.obj)/compile.cmd : FORCE | tool-ar tool-fc
	@mkdir -p $(@D)
	@$(call record_command,$(FCOMPILE),$@)

$(refblas.objects) : $(refblas.obj)/%.o : refblas/%.f $(refblas.obj)/compile.cmd
	$(FCOMPILE) -c -o $@ $<

all : cblas clapack

cblas     : $(top.cblas.all)
atlblas   : $(top.atl.all)
clapack   : $(top.cblas.all) $(top.clapack.all)
atllapack : $(top.atl.all) $(top.atllapack.all)
f77blas   : $(top.f77.all)
f77lapack : $(top.f77.all) $(top.f77lapack.all)
refblas   : $(refblas.lib)
lbt       : $(lbt.lib)

#
#  The shared library with the standard interfaces (mk/shared.mk):
#  libulmblas.so / .dylib / .dll; SHARED_VERSION=0: with the version in the
#  name (libulmblas.so.0, libulmblas.0.dylib) and a link without it.
#  INTERFACE64=1 (or make shared64): libulmblas64, the interfaces with
#  64-bit integers under the usual names.  ILP64=0: without the interfaces
#  with 64-bit integers and the suffix _64_.
#
#  (build-shared and install-shared, mk/install.mk: on platforms without
#  the dispatching library the version is 0)
ifeq ($(origin SHARED_VERSION),undefined)
SHARED_VERSION := $(if $(dispatch.platform),,$(if $(filter build-shared \
                      install-shared check-package,$(MAKECMDGOALS)),0))
endif
INTERFACE64    ?= 0

top.shared.dir     := interfaces/blas/shared/build.shared
top.shared.out     := .
top.shared.name    := ulmblas
top.shared.version  = $(SHARED_VERSION)
top.shared.i64     := 0
top.shared.ilp64    = $(ILP64)
top.shared.arch     = $(ARCHFLAGS)
top.shared.sonames := $(if $(dispatch.platform),,$(if \
                          $(filter elf,$(shared.platform)),libblas.so.3 libcblas.so.3))

#  their rules only for the goals that need them: on Windows the
#  dispatching library has the same name (libulmblas.dll), and the
#  installation needs them only on platforms without backends
top.shared.goals := shared shared64 $(if $(dispatch.platform),,sonames \
                    build-shared install-shared check-package)
ifneq ($(filter $(top.shared.goals),$(MAKECMDGOALS)),)
$(eval $(call shared_lib,top.shared))

top.shared64.dir     := interfaces/blas/shared/build.shared64
top.shared64.out     := .
top.shared64.name    := ulmblas64
top.shared64.version  = $(SHARED_VERSION)
top.shared64.i64     := 1
top.shared64.ilp64   := 0
top.shared64.arch     = $(ARCHFLAGS)
$(eval $(call shared_lib,top.shared64))
endif

shared   : $(if $(filter 1,$(INTERFACE64)),$(top.shared64.lib),$(top.shared.lib))
shared64 : $(top.shared64.lib)

#
#  Apple Silicon and x86-64: libulmblas selects a backend (libulmblas_sme,
#  _neon; _avx512, _avx2, _sse3, _generic) at load time (mk/dispatch.mk).
#  sonames (Linux): the same library as libblas.so.3 and libcblas.so.3,
#  the SONAMEs (shared object names) of the BLAS of the distributions
#  (BACKEND_DIR: where the backends are installed); on platforms without
#  backends the plain library linked with these names.
#
top.dispatch.dir := interfaces/blas/dispatch/build.dispatch
top.dispatch.out := .

#  its rules only for the goals that need it (make shared SHARED_VERSION=0
#  makes a library of the same name)
top.dispatch.goals := shared-dispatch sonames build-shared install-shared \
                      check-package

ifneq ($(dispatch.platform),)
ifneq ($(filter $(top.dispatch.goals),$(MAKECMDGOALS)),)
$(eval $(call dispatch_lib,top.dispatch))
endif
shared-dispatch : $(top.dispatch.all)
sonames : $(top.dispatch.sonames) $(top.dispatch.backend_libs)
else
shared-dispatch :
	@echo "error: no dispatching library for $(TARGET_MACHINE): use make shared SHARED_VERSION=0"; exit 2
sonames : $(addprefix $(top.shared.out)/,$(top.shared.sonames))
endif

include $(TOPDIR)/mk/install.mk

#
#  The Netlib test programs (test/F77, test/C) against libulmblas (needs
#  FC), built in its own directory (test/build/shared-lib): in the top
#  directory libulmblas.dylib (.so) may be the link of make shared-dispatch
#  to the dispatching library.  Then the interfaces with 64-bit integers
#  (_64_ names, and libulmblas64).
#
ifneq ($(filter check-shared,$(MAKECMDGOALS)),)
chk.shared.dir     := test/build/shared-lib/build
chk.shared.out     := test/build/shared-lib
chk.shared.name    := ulmblas
chk.shared.version  = $(SHARED_VERSION)
chk.shared.i64     := 0
chk.shared.ilp64    = $(ILP64)
chk.shared.arch     = $(ARCHFLAGS)
$(eval $(call shared_lib,chk.shared))

chk.shared64.dir     := test/build/shared-lib/build64
chk.shared64.out     := test/build/shared-lib
chk.shared64.name    := ulmblas64
chk.shared64.version  = $(SHARED_VERSION)
chk.shared64.i64     := 1
chk.shared64.ilp64   := 0
chk.shared64.arch     = $(ARCHFLAGS)
$(eval $(call shared_lib,chk.shared64))
endif

check-shared : $(chk.shared.lib) $(chk.shared64.lib) | tool-ar
	FC="$(FC)" CC="$(CC)" $(SHELL) test/check-shared.sh test/build/shared-lib \
	         test/build/shared-lib/build test/build/shared
	$(if $(filter 0,$(ILP64)),,FC="$(FC)" CC="$(CC)" NM="$(NM)" \
	    $(SHELL) test/check-ilp64.sh test/build/shared-lib test/build/ilp64)
	FC="$(FC)" CC="$(CC)" NM="$(NM)" $(SHELL) test/check-ilp64.sh \
	    test/build/shared-lib test/build/plain64 ulmblas64 - \
	    test/build/shared-lib/build64

#
#  The same test programs against the dispatching library (built in its own
#  directory, test/build/dispatch-lib), without the CBLAS (C interface to
#  the BLAS) programs of levels
#  2 and 3 (see test/check-shared.sh), and c_scal with each backend the CPU
#  (central processing unit)
#  supports
#
ifneq ($(filter check-dispatch,$(MAKECMDGOALS)),)
ifneq ($(dispatch.platform),)
chk.dispatch.dir := test/build/dispatch-lib/build
chk.dispatch.out := test/build/dispatch-lib
$(eval $(call dispatch_lib,chk.dispatch))
endif
endif

check-dispatch : $(chk.dispatch.all) | tool-ar shared-dispatch-platform
	inc=`ls test/build/dispatch-lib/build/*/cblas.h | head -1` && \
	FC="$(FC)" CC="$(CC)" $(SHELL) test/check-shared.sh test/build/dispatch-lib \
	         `dirname $$inc` test/build/dispatch ulmblas f77
	if [ -f test/build/dispatch-lib/build/interface64/include/cblas.h ]; then \
	    FC="$(FC)" CC="$(CC)" NM="$(NM)" $(SHELL) test/check-ilp64.sh \
	        test/build/dispatch-lib test/build/dispatch64 ulmblas64 front \
	        test/build/dispatch-lib/build/interface64/include; \
	fi

.PHONY : shared-dispatch-platform
shared-dispatch-platform :
	@test -n "$(dispatch.platform)" || \
	    { echo "error: no dispatching library for $(TARGET_MACHINE)"; exit 2; }

#
#  Tests (mk/test.mk) and the full check with the ATLAS testers and the
#  benchmarks (mk/bench.mk)
#
include $(TOPDIR)/mk/test.mk
include $(TOPDIR)/mk/bench.mk

#  The synchronization of the parallel gemm frame, all interleavings of
#  small cases (test/protocol/frame.py; FULL=1: larger cases, about a minute)
check-protocol :
	python3 test/protocol/frame.py $(if $(filter 1,$(FULL)),--full)

#
#  ARCHFLAGS configurations this machine and CXX support, the micro kernels
#  they select and a suggested one (mk/check_arch.sh).  Builds nothing.
#
check_arch :
	@CXX='$(CXX)' ARCHFLAGS='$(ARCHFLAGS)' ARCHFLAGS_ORIGIN='$(origin ARCHFLAGS)' \
	 ARCHFLAGS_DEFAULT='$(ULMBLAS_ARCHFLAGS_DEFAULT)' \
	 $(SHELL) $(TOPDIR)/mk/check_arch.sh

#
#  GEMM (general matrix-matrix product) benchmark of ulmBLAS against the
#  BLAS libraries found on this machine (bench/gemm/README.md): builds
#  libcblas.a with ARCHFLAGS (which select the micro kernels), then runs
#  bench/gemm/run.sh.  Settings: PREC,
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

#  (the shared libraries by name: their rules are made only for the goals
#  that build them)
clean :
	rm -rf $(sort $(lib.clean)) $(test.clean) $(bench.clean) \
	       interfaces/blas/shared/build.shared \
	       interfaces/blas/shared/build.shared64 \
	       interfaces/blas/dispatch/build.dispatch \
	       libulmblas.so libulmblas.so.0 libulmblas.dylib libulmblas.0.dylib \
	       libulmblas.dll libulmblas.dll.a ulmblas.lib \
	       libulmblas64.so libulmblas64.so.0 libulmblas64.dylib \
	       libulmblas64.0.dylib libulmblas_*.so libulmblas_*.dylib \
	       libulmblas_*.dll libulmblas_*.dll.a libblas.so.3 libcblas.so.3

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
	@echo '  shared64         libulmblas64: the interfaces with 64-bit integers under'
	@echo '                   the usual names'
	@echo '  shared-dispatch  Apple Silicon and x86-64: libulmblas selects a backend'
	@echo '                   (sme, neon; avx512, avx2, sse3, generic) at load time'
	@echo '  build-shared     the libraries install-shared installs (PREFIX=dir,'
	@echo '                   see mk/install.mk; libulmlapack needs FC)'
	@echo '  install-shared   install them in PREFIX (DESTDIR, ULMBLAS_LIBDIR, ...)'
	@echo '  check-package    Netlib test programs against the libraries of'
	@echo '                   build-shared (needs FC)'
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
	@echo '                   (PREC=s|d|c|z LIBS=auto|"ulmblas openblas ..." SIZES=...,'
	@echo '                   see bench/gemm/README.md)'
	@echo '  tune             measure GEMM on this machine, generate its selection table'
	@echo '  tune-params      measure threads, variants, block sizes (ulmblas/config/tuned.h)'
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

-include $(lib.depfiles)
