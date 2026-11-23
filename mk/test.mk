#
#  Tests of ulmBLAS:
#
#      make check           all tests
#      make check-numeric   only the C++ tests (no Fortran compiler needed)
#      make check-netlib    only the Netlib BLAS (Basic Linear Algebra
#                           Subprograms) test suites (FC, CC needed)
#      make check_cblas, check_f77blas
#                           only the Netlib suite of CBLAS (C interface to
#                           the BLAS) or Fortran 77
#      make check-reference-lapack
#                           test programs of Reference-LAPACK 3.12.1, the
#                           reference implementation of LAPACK (Linear
#                           Algebra PACKage) (downloaded, not part of make
#                           check)
#      make check-reference-lapack-lin
#                           LU (lower-upper) and Cholesky (paths GE, PO)
#                           with the test
#                           programs of Reference-LAPACK 3.12.1 for linear
#                           equations (downloaded, not part of make check)
#      make check-conformance
#                           differential test against the reference BLAS
#                           and CBLAS of LAPACK 3.12.1 (downloaded, not
#                           part of make check)
#
#  All tests are built and run for each test configuration in test/config/
#  (different block sizes, see test/config/make.testcase*.inc) in
#  test/build/<configuration>:
#
#      lib/          libcblas.a, libclapack.a, libf77blas.a and headers
#      obj-*/        object files of the libraries
#      numeric/      numeric_tests (C++ tests, test/numeric/*.cc)
#      examples/     the C++ examples of examples/ (header only)
#      netlib-C/     Netlib CBLAS test programs (C and Fortran, test/C/*)
#      netlib-F77/   Netlib BLAS test programs (Fortran, test/F77/*)
#
#  The tests are run and evaluated by test/run-tests.sh.  make check runs
#  the Netlib test suites only if a usable Fortran compiler, C compiler and
#  Fortran runtime are found; otherwise they are reported as SKIP with the
#  reason.  With STRICT=1 skipped tests make 'make check' fail.
#
#  Variables:
#
#      CONFIGS        test configurations   (default: testcase*, the files
#                                           test/config/make.testcase*.inc)
#      NETLIB_SUITES  Netlib test suites    (default: C F77)
#      TESTS          only run tests whose ID (identifier) or name contains
#                     one of these
#                     strings
#      VERBOSE=1      print every test result, not only problems
#      STRICT=1       skipped tests are errors
#
CONFIGS       ?= $(patsubst test/config/make.%.inc,%,\
                            $(wildcard test/config/make.testcase*.inc))
NETLIB_SUITES ?= C F77
REFERENCE_CONFIG ?= testcase04

ifneq ($(filter check_cblas,$(MAKECMDGOALS)),)
NETLIB_SUITES := C
endif
ifneq ($(filter check_f77blas,$(MAKECMDGOALS)),)
NETLIB_SUITES := F77
endif

.PHONY : build-numeric build-netlib

test.clean := test/build

#  the configurations the goals need (their rules are made only then)
test.goals.all := check check-numeric check-netlib check_cblas check_f77blas
test.goals.ref := check-reference-lapack check-reference-lapack-lin \
                  check-conformance
test.configs := $(sort $(if $(filter $(test.goals.all),$(MAKECMDGOALS)),$(CONFIGS)) \
                       $(if $(filter $(test.goals.ref),$(MAKECMDGOALS)),$(REFERENCE_CONFIG)))

RUN_TESTS = TEST_BUILD='build' CONFIGS='$(CONFIGS)' \
            TESTS='$(TESTS)' VERBOSE='$(VERBOSE)' STRICT='$(STRICT)' \
            CC=$(call shell_quote,$(CC) $(CPPFLAGS) $(ULMBLAS_CFLAGS) $(CFLAGS)) \
            CXX=$(call shell_quote,$(CXX) $(CXXSTD) $(CPPFLAGS)) \
            $(SHELL) run-tests.sh

#
#  A configuration can require CPU (central processing unit) features
#  (CONFIG_REQUIRES, e.g. avx
#  fma, or neon).  It is only built if the compiler accepts its flags and
#  the CPU and the operating system support the features (mk/probe.sh
#  cpu); otherwise its targets only write the reason to
#  test/build/CONFIG/unsupported, and run-tests.sh reports the tests of the
#  configuration as SKIP.  An error of the probe itself (exit status 2, e.g.
#  the compiler does not work) stops the build: it is not a reason to skip
#  the tests.
#
test_probe = $(shell \
    out=`PROBE_CXX_BASE='$(CXX) $(ULMBLAS_CXXFLAGS) $(CXXFLAGS)' \
         PROBE_CXX='$(CXX) $(test.$1.cppflags) $(ULMBLAS_CXXFLAGS) $(CXXFLAGS)' \
         PROBE_LD='$(LDFLAGS) $(LDLIBS)' PROBE_FEATURES='$(test.$1.requires)' \
         $(SHELL) $(TOPDIR)/mk/probe.sh cpu 2>&1 > /dev/null`; \
    echo $$? `echo "$$out" | sed -n 's/^error: //p' | head -1`)

#
#  Netlib CBLAS test suite: Fortran drivers calling the C interface through
#  C wrappers.  The drivers are compiled with -fallow-argument-mismatch if
#  the Fortran compiler supports it (required by gfortran 10 or newer for
#  c_[sd]blat1.f).  The C wrappers pass pointers to complex structures to
#  the C interface, which expects float/double pointers, hence
#  -Wno-incompatible-pointer-types (an error by default in GCC (GNU
#  Compiler Collection) 14 or newer).
#
NETLIB_FFLAGS  = $(eval NETLIB_FFLAGS := $(shell \
                     echo | $(FC) -fallow-argument-mismatch -x f77 -c - \
                          -o /dev/null >/dev/null 2>&1 \
                     && echo -fallow-argument-mismatch) -w)$(NETLIB_FFLAGS)

#  T0351 calls the library from several threads (std::thread; -pthread is
#  needed with older C libraries).
NUMERIC_THREADS = -pthread
NUMERIC_DEFS    = -D'BLAS_HEADER=<cblas.h>' -D'LAPACK_HEADER=<clapack.h>' \
                  -D'ULMLAPACK(x)=ulmblas_ $(PASTE) x'

#  the compile commands of a configuration C (FFLAGS and the index type
#  may be its own)
test_cxxcompile = $(CXX) -D'ULMBLAS_INDEX_TYPE_OF_BUILD=$(test.$1.index)' \
                  $(test.$1.cppflags) $(CPPFLAGS) $(ULMBLAS_CXXFLAGS) \
                  $(CXXFLAGS) $(DEPFLAGS)
test_fcompile   = $(FC) $(FOPTFLAGS) $(test.$1.fflags)

#
#  $(eval $(call test_read_config,C)): the settings of configuration C
#  (test/config/make.C.inc) in test.C.*; the variables the file sets are
#  reset before and restored after
#
define test_read_config
CONFIG_CPPFLAGS      :=
CONFIG_REQUIRES      :=
CONFIG_NETLIB_C_SKIP :=
test.saved.index     := $$(ULMBLAS_INDEX_TYPE)
test.saved.fflags    := $$(FFLAGS)
include test/config/make.$1.inc
test.$1.cppflags      := $$(CONFIG_CPPFLAGS)
test.$1.requires      := $$(CONFIG_REQUIRES)
test.$1.netlib_c_skip := $$(CONFIG_NETLIB_C_SKIP)
test.$1.index         := $$(ULMBLAS_INDEX_TYPE)
test.$1.fflags        := $$(FFLAGS)
ULMBLAS_INDEX_TYPE    := $$(test.saved.index)
FFLAGS                := $$(test.saved.fflags)
endef

#
#  $(eval $(call test_config,C)): the libraries and test programs of
#  configuration C
#
define test_config
$$(eval $$(call test_read_config,$1))
test.$1.b := test/build/$1
test.$1.lib := test/build/$1/lib
ifneq ($$(strip $$(test.$1.requires)),)
test.$1.probe := $$(call test_probe,$1)
ifeq ($$(firstword $$(test.$1.probe)),1)
test.$1.unsupported := $$(wordlist 2,$$(words $$(test.$1.probe)),$$(test.$1.probe))
else ifneq ($$(firstword $$(test.$1.probe)),0)
$$(error $1: probe of the CPU features '$$(test.$1.requires)' failed \
         (exit status $$(test.$1.probe)))
endif
endif

ifneq ($$(test.$1.unsupported),)
.PHONY : test.$1.unsupported
test.$1.unsupported :
	@mkdir -p $$(test.$1.b)
	@echo '$$(test.$1.unsupported)' > $$(test.$1.b)/unsupported
test.$1.libs       := test.$1.unsupported
test.$1.libs-f77   := test.$1.unsupported
test.$1.numeric    := test.$1.unsupported
test.$1.netlib-C   := test.$1.unsupported
test.$1.netlib-F77 := test.$1.unsupported
else

#  the libraries
test.$1.cblas.out      := $$(test.$1.lib)
test.$1.cblas.obj      := $$(test.$1.b)/obj-cblas
test.$1.cblas.index    := $$(test.$1.index)
test.$1.cblas.config   := $$(test.$1.cppflags)
test.$1.cblas.asm      := yes
test.$1.cblas.tools    := tool-ar tool-cxx-link
$$(eval $$(call cblas_lib,test.$1.cblas,cblas))

test.$1.clapack.out    := $$(test.$1.lib)
test.$1.clapack.obj    := $$(test.$1.b)/obj-clapack
test.$1.clapack.index  := $$(test.$1.index)
test.$1.clapack.config := $$(test.$1.cppflags)
test.$1.clapack.asm    := yes
test.$1.clapack.blas   := test.$1.cblas
test.$1.clapack.tools  := tool-ar tool-cxx-link
$$(eval $$(call clapack_lib,test.$1.clapack,cblas))

test.$1.f77.out        := $$(test.$1.lib)
test.$1.f77.obj        := $$(test.$1.b)/obj-f77blas
test.$1.f77.index      := $$(test.$1.index)
test.$1.f77.config     := $$(test.$1.cppflags)
test.$1.f77.fflags     := $$(test.$1.fflags)
test.$1.f77.asm        := yes
test.$1.f77.tools      := tool-ar
$$(eval $$(call f77blas_lib,test.$1.f77,f77))

test.$1.f77lapack.out       := $$(test.$1.lib)
test.$1.f77lapack.obj       := $$(test.$1.b)/obj-f77lapack
test.$1.f77lapack.index     := $$(test.$1.index)
test.$1.f77lapack.config    := $$(test.$1.cppflags)
test.$1.f77lapack.blas      := test.$1.f77
test.$1.f77lapack.guardname := $1
test.$1.f77lapack.tools     := tool-ar
$$(eval $$(call f77lapack_lib,test.$1.f77lapack,f77))

#  a configuration that was unsupported before
.PHONY : test.$1.supported
test.$1.supported :
	@rm -f $$(test.$1.b)/unsupported

test.$1.libs     := $$(test.$1.cblas.all) $$(test.$1.clapack.all) test.$1.supported
test.$1.libs-f77 := $$(test.$1.f77.all) $$(test.$1.f77lapack.all)

#  Numeric tests (C++).  The objects depend on the generated headers
#  through the dependency files; the libraries are order-only
#  prerequisites (they must exist, but a rebuilt library does not require
#  recompiling the tests).
test.$1.numeric.o := $$(patsubst test/numeric/%.cc,$$(test.$1.b)/numeric/%.o,\
                                $$(wildcard test/numeric/*.cc))
test.$1.numeric.x := $$(test.$1.b)/numeric/numeric_tests
test.$1.examples  := $$(patsubst examples/%.cc,$$(test.$1.b)/examples/%,\
                                $$(wildcard examples/*.cc))
test.$1.numeric   := $$(test.$1.numeric.x) $$(test.$1.examples) test.$1.supported
lib.depfiles      += $$(test.$1.numeric.o:.o=.d)

$$(test.$1.b)/numeric/compile.cmd : FORCE | tool-ar tool-cxx-link
	@mkdir -p $$(@D)
	@$$(call record_command,$$(call test_cxxcompile,$1) $$(NUMERIC_THREADS) $$(NUMERIC_DEFS) ; $$(LDFLAGS) $$(LDLIBS),$$@)

$$(test.$1.numeric.o) : $$(test.$1.b)/numeric/%.o : test/numeric/%.cc \
        $$(test.$1.b)/numeric/compile.cmd | $$(test.$1.clapack.lib)
	$$(call test_cxxcompile,$1) $$(NUMERIC_THREADS) $$(NUMERIC_DEFS) \
	    -I$$(test.$1.lib) -I$$(TOPDIR) -c -o $$@ $$<

$$(test.$1.numeric.x) : $$(test.$1.numeric.o) $$(test.$1.clapack.lib) \
                        $$(test.$1.cblas.lib)
	$$(CXX) $$(ULMBLAS_CXXFLAGS) $$(CXXFLAGS) $$(NUMERIC_THREADS) $$(LDFLAGS) \
	       -o $$@ $$(test.$1.numeric.o) $$(test.$1.clapack.lib) \
	       $$(test.$1.cblas.lib) $$(LDLIBS) $$(OMPLIBS)

#  Examples: the C++ examples of the README (examples/*.cc), compiled with
#  the flags of the configuration (its kernels and block sizes) and run by
#  run-tests.sh.  They are header only, but a selected assembly kernel
#  (OpenBLAS's HASWELL DGEMM kernel) is an object file, which they take
#  from the BLAS library.
$$(test.$1.examples) : $$(test.$1.b)/examples/% : examples/%.cc \
                       | $$(test.$1.cblas.lib)
	@mkdir -p $$(@D)
	$$(call test_cxxcompile,$1) -I$$(TOPDIR) $$(LDFLAGS) -o $$@ $$< \
	    $$(test.$1.cblas.lib) $$(LDLIBS)

#  Netlib CBLAS test suite.  c_xblatN: Fortran driver c_xblatN.f and C
#  wrappers c_xblasN.c, for levels 2 and 3 also c_xNchke.c and the
#  auxiliary functions.  Fortran objects are linked with the C++ compiler
#  and FLIBS (Fortran runtime).  A configuration can skip them
#  (CONFIG_NETLIB_C_SKIP = reason, e.g. their int arguments do not match a
#  64-bit ULMBLAS_INDEX_TYPE): run-tests.sh reports them as SKIP.
test.$1.nc := $$(test.$1.b)/netlib-C
test.$1.nc.x := $$(foreach t,s d c z,$$(foreach l,1 2 3,$$(test.$1.nc)/c_$$(t)blat$$(l)))
test.$1.netlib-C := test.$1.netlib-C
ifneq ($$(strip $$(test.$1.netlib_c_skip)),)
.PHONY : test.$1.netlib-C
test.$1.netlib-C :
	@mkdir -p $$(test.$1.b)
	@echo '$$(test.$1.netlib_c_skip)' > $$(test.$1.b)/netlib-C.skip
else
.PHONY : test.$1.netlib-C
test.$1.netlib-C : $$(test.$1.nc.x) test.$1.supported
	@rm -f $$(test.$1.b)/netlib-C.skip
endif

$$(test.$1.nc)/compile.cmd : FORCE | tool-ar tool-cc tool-fc-link
	@mkdir -p $$(@D)
	@$$(call record_command,$$(CC) $$(ULMBLAS_CFLAGS) $$(CFLAGS) ; $$(call test_fcompile,$1) $$(NETLIB_FFLAGS) ; $$(CXX) $$(LDFLAGS) $$(LDLIBS) $$(FLIBS),$$@)

$$(test.$1.nc)/%.o : test/C/%.c $$(test.$1.cblas.header) \
                     $$(test.$1.nc)/compile.cmd | $$(test.$1.cblas.lib)
	$$(CC) $$(CPPFLAGS) $$(ULMBLAS_CFLAGS) $$(CFLAGS) -w \
	    -Wno-incompatible-pointer-types -DADD_ -D'BLAS_HEADER=<cblas.h>' \
	    -I$$(test.$1.lib) -I$$(TOPDIR) -c -o $$@ $$<

$$(test.$1.nc)/%.o : test/C/%.f $$(test.$1.nc)/compile.cmd
	$$(call test_fcompile,$1) $$(NETLIB_FFLAGS) -c -o $$@ $$<

$$(foreach t,s d c z,$$(foreach l,1 2 3,\
    $$(eval $$(call test_cblas_program,$1,$$(t),$$(l)))))

#  Netlib BLAS test suite for the Fortran 77 interface
test.$1.nf := $$(test.$1.b)/netlib-F77
test.$1.nf.x := $$(foreach p,$$(foreach t,s d c z,$$(foreach l,1 2 3,$$(t)blat$$(l))) ublat1,\
                    $$(test.$1.nf)/$$(p))
test.$1.nf.dat := $$(patsubst test/F77/%.dat.in,$$(test.$1.nf)/%.dat,\
                              $$(wildcard test/F77/*.dat.in))
test.$1.netlib-F77 := $$(test.$1.nf.x) $$(test.$1.nf.dat) test.$1.supported

$$(test.$1.nf)/compile.cmd : FORCE | tool-ar tool-fc-link
	@mkdir -p $$(@D)
	@$$(call record_command,$$(call test_fcompile,$1) $$(NETLIB_FFLAGS) ; $$(CXX) $$(LDFLAGS) $$(LDLIBS) $$(FLIBS),$$@)

$$(test.$1.nf)/%.o : test/F77/%.f $$(test.$1.nf)/compile.cmd
	$$(call test_fcompile,$1) $$(NETLIB_FFLAGS) -c -o $$@ $$<

$$(test.$1.nf.x) : $$(test.$1.nf)/% : $$(test.$1.nf)/%.o $$(test.$1.f77.lib)
	$$(CXX) $$(ULMBLAS_CXXFLAGS) $$(CXXFLAGS) $$(LDFLAGS) -o $$@ $$< \
	       $$(test.$1.f77.lib) $$(LDLIBS) $$(FLIBS)

$$(test.$1.nf.dat) : $$(test.$1.nf)/%.dat : test/F77/%.dat.in
	@mkdir -p $$(@D)
	cp $$< $$@
endif
endef

#  $(call test_cblas_program,C,T,L): the program c_TblatL of configuration C
define test_cblas_program
$$(test.$1.nc)/c_$2blat$3 : $$(test.$1.nc)/c_$2blat$3.o $$(test.$1.nc)/c_$2blas$3.o \
        $$(if $$(filter 1,$3),,$$(test.$1.nc)/c_$2$3chke.o \
              $$(test.$1.nc)/auxiliary.o $$(test.$1.nc)/c_xerbla.o) \
        $$(test.$1.cblas.lib)
	$$(CXX) $$(ULMBLAS_CXXFLAGS) $$(CXXFLAGS) $$(LDFLAGS) -o $$@ \
	        $$(filter %.o,$$^) $$(test.$1.cblas.lib) $$(LDLIBS) $$(FLIBS)
endef

$(foreach c,$(test.configs),$(eval $(call test_config,$c)))

#
#  check: the Netlib suites are included if the tools they need work
#  (checked when make reads this file; the reason is reported by
#  run-tests.sh)
#
test.probe_log := test/build/netlib-tools.log
ifneq ($(filter check,$(MAKECMDGOALS)),)
test.netlib.ok := $(shell mkdir -p test/build && \
    { $(PROBE) fc > $(test.probe_log) 2>&1 \
      && $(PROBE) cc >> $(test.probe_log) 2>&1 \
      && $(PROBE) fc-link >> $(test.probe_log) 2>&1; } \
    && rm -f $(test.probe_log) && echo yes)
endif

build-numeric : $(foreach c,$(CONFIGS),$(test.$c.numeric))

build-netlib : $(foreach s,$(NETLIB_SUITES),$(foreach c,$(CONFIGS),$(test.$c.netlib-$s)))

check : build-numeric $(if $(test.netlib.ok),build-netlib) | tool-ar
	@$(if $(test.netlib.ok),,echo "note: Netlib test suites were skipped, see below";) \
	cd test && NUMERIC=yes \
	    NETLIB_SUITES='$(if $(test.netlib.ok),$(NETLIB_SUITES))' \
	    NETLIB_SKIP_LOG=build/netlib-tools.log $(RUN_TESTS)

check-numeric : build-numeric
	@cd test && NUMERIC=yes NETLIB_SUITES= $(RUN_TESTS)

check-netlib check_cblas check_f77blas : build-netlib | tool-ar tool-cc tool-fc-link
	@cd test && NUMERIC=no NETLIB_SUITES='$(NETLIB_SUITES)' $(RUN_TESTS)

#
#  check-reference-lapack: the BLAS and CBLAS test programs of
#  Reference-LAPACK 3.12.1, downloaded (network access needed) and run
#  against the libraries of configuration REFERENCE_CONFIG (default block
#  sizes and kernels), see test/reference-lapack.sh.  Not part of make
#  check.
#
test.ref.libs = $(test.$(REFERENCE_CONFIG).libs) $(test.$(REFERENCE_CONFIG).libs-f77)

check-reference-lapack : $(test.ref.libs) | tool-ar tool-cc tool-fc-link
	@cd test && CC=$(call shell_quote,$(CC)) CFLAGS=$(call shell_quote,$(CPPFLAGS) $(ULMBLAS_CFLAGS) $(CFLAGS)) \
	 FC=$(call shell_quote,$(FC)) FFLAGS=$(call shell_quote,$(FOPTFLAGS) $(FFLAGS)) \
	 CXX=$(call shell_quote,$(CXX) $(ULMBLAS_CXXFLAGS) $(CXXFLAGS)) \
	 LDFLAGS=$(call shell_quote,$(LDFLAGS)) LDLIBS=$(call shell_quote,$(LDLIBS)) \
	 FLIBS=$(call shell_quote,$(FLIBS)) \
	 $(SHELL) reference-lapack.sh build/$(REFERENCE_CONFIG)/lib \
	          build/reference-lapack

#
#  check-reference-lapack-lin: the LAPACK functions of ulmBLAS (Fortran 77
#  interface) with the test programs xlintst[sdcz] of Reference-LAPACK
#  3.12.1 on the paths GE, PO, TR and S2/H2, see
#  test/reference-lapack-lin.sh.  Not part of make check.
#
check-reference-lapack-lin : $(test.$(REFERENCE_CONFIG).libs-f77) | tool-ar tool-fc-link
	@cd test && FC=$(call shell_quote,$(FC)) FFLAGS=$(call shell_quote,$(FOPTFLAGS)) \
	 $(if $(LIN_LDLIBS),LDLIBS=$(call shell_quote,$(LIN_LDLIBS))) \
	 $(if $(wildcard test/build/reference-lapack/lapack-3.12.1.tar.gz),LAPACK_TARBALL=build/reference-lapack/lapack-3.12.1.tar.gz) \
	 $(SHELL) reference-lapack-lin.sh build/$(REFERENCE_CONFIG)/lib \
	          build/reference-lapack-lin

#
#  check-conformance: every level 1, 2 and 3 routine of the Fortran 77 and
#  C interfaces of configuration REFERENCE_CONFIG (index type int)
#  compared call by call with the reference BLAS and CBLAS of LAPACK
#  3.12.1, built from the downloaded sources (see
#  test/conformance/conformance.sh and blas_conformance.md).  Not part of
#  make check.
#
check-conformance : $(test.ref.libs) | tool-ar tool-cc tool-fc-link
	@cd test && CC=$(call shell_quote,$(CC)) CFLAGS=$(call shell_quote,$(CPPFLAGS) $(ULMBLAS_CFLAGS) $(CFLAGS)) \
	 FC=$(call shell_quote,$(FC)) FFLAGS=$(call shell_quote,$(FOPTFLAGS) $(FFLAGS)) \
	 CXX=$(call shell_quote,$(CXX) $(ULMBLAS_CXXFLAGS) $(CXXFLAGS)) \
	 LDFLAGS=$(call shell_quote,$(LDFLAGS)) LDLIBS=$(call shell_quote,$(LDLIBS)) \
	 FLIBS=$(call shell_quote,$(FLIBS)) \
	 $(SHELL) conformance/conformance.sh build/$(REFERENCE_CONFIG)/lib \
	          build/reference-lapack
