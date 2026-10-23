#
#  Tests of ulmBLAS.  Usually invoked from the top-level directory:
#
#      make check           all tests
#      make check-numeric   only the C++ tests (no Fortran compiler needed)
#      make check-netlib    only the Netlib BLAS test suites (FC, CC needed)
#
#  All tests are built and run for each test configuration in config/
#  (different block sizes, see config/make.testcase*.inc) in
#  build/<configuration>.  The tests are run and evaluated by run-tests.sh.
#
#  make check runs the Netlib test suites only if a usable Fortran compiler,
#  C compiler and Fortran runtime are found; otherwise they are reported as
#  SKIP with the reason.  With STRICT=1 skipped tests make 'make check' fail.
#
#  Variables:
#
#      CONFIGS        test configurations   (default: all in config/)
#      NETLIB_SUITES  Netlib test suites    (default: C F77)
#      TESTS          only run tests whose ID or name contains one of these
#                     strings
#      VERBOSE=1      print every test result, not only problems
#      STRICT=1       skipped tests are errors
#
TOPDIR = ..

include $(TOPDIR)/make.inc

CONFIGS       ?= $(patsubst config/make.%.inc,%,\
                            $(wildcard config/make.testcase*.inc))
NETLIB_SUITES ?= C F77

LIB_TARGETS     = $(addprefix lib-,$(CONFIGS))
LIBF77_TARGETS  = $(addprefix lib-f77-,$(CONFIGS))
NUMERIC_TARGETS = $(addprefix numeric-,$(CONFIGS))
NETLIB_TARGETS  = $(foreach s,$(NETLIB_SUITES),$(addprefix netlib-$(s)-,$(CONFIGS)))

BUILD          = build
PROBE_LOG      = $(BUILD)/netlib-tools.log

RUN_TESTS      = TEST_BUILD='$(BUILD)' CONFIGS='$(CONFIGS)' \
                 TESTS='$(TESTS)' VERBOSE='$(VERBOSE)' STRICT='$(STRICT)' \
                 CC=$(call shell_quote,$(CC) $(CPPFLAGS) $(CFLAGS)) \
                 $(SHELL) run-tests.sh

.PHONY : all check check-numeric check-netlib build-numeric build-netlib \
         clean $(LIB_TARGETS) $(LIBF77_TARGETS) $(NUMERIC_TARGETS) \
         $(NETLIB_TARGETS)

all : check

include $(TOPDIR)/mk/tools.mk

#
#  Build targets (one per configuration and kind of test, so that the
#  configurations can be built in parallel with make -j)
#
build-numeric : $(NUMERIC_TARGETS)

build-netlib : $(NETLIB_TARGETS)

$(LIB_TARGETS) : lib-% : tool-ar
	@$(MAKE) -f Makefile.config CONFIG=$* libs

$(LIBF77_TARGETS) : lib-f77-% : tool-ar
	@$(MAKE) -f Makefile.config CONFIG=$* libs-f77

$(NUMERIC_TARGETS) : numeric-% : lib-%
	@$(MAKE) -f Makefile.config CONFIG=$* numeric

$(filter netlib-C-%,$(NETLIB_TARGETS)) : netlib-C-% : lib-%
	@$(MAKE) -f Makefile.config CONFIG=$* netlib-C

$(filter netlib-F77-%,$(NETLIB_TARGETS)) : netlib-F77-% : lib-f77-%
	@$(MAKE) -f Makefile.config CONFIG=$* netlib-F77

#
#  check: the Netlib suites are included if the tools they need work.
#
check : tool-ar
	@mkdir -p $(BUILD); \
	if  $(PROBE) fc > $(PROBE_LOG) 2>&1 \
	 && $(PROBE) cc >> $(PROBE_LOG) 2>&1 \
	 && $(PROBE) fc-link >> $(PROBE_LOG) 2>&1; \
	then \
	    suites='$(NETLIB_SUITES)'; rm -f $(PROBE_LOG); \
	    targets='build-numeric build-netlib'; \
	else \
	    suites=; targets=build-numeric; \
	    echo "note: Netlib test suites will be skipped, see below"; \
	fi; \
	$(MAKE) $$targets \
	&& NUMERIC=yes NETLIB_SUITES="$$suites" NETLIB_SKIP_LOG=$(PROBE_LOG) \
	   $(RUN_TESTS)

check-numeric : build-numeric
	@NUMERIC=yes NETLIB_SUITES= $(RUN_TESTS)

check-netlib : tool-ar tool-cc tool-fc-link
	@$(MAKE) build-netlib
	@NUMERIC=no NETLIB_SUITES='$(NETLIB_SUITES)' $(RUN_TESTS)

clean :
	rm -rf $(BUILD)
