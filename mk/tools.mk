#
#  Usability checks of the tools needed by the build.  Each check is a phony
#  target that compiles, links and runs a tiny program with the configured
#  tool and flags (see mk/probe.sh).  If that fails, the error explains what
#  the tool is needed for and how to select another one.
#
#      tool-cxx      C++ compiler           (libraries, C++ tests)
#      tool-ar       archiver and ranlib    (static libraries)
#      tool-cc       C compiler             (Netlib CBLAS tests, benchmarks)
#      tool-fc       Fortran compiler       (F77 interface, refblas, Netlib
#                                            tests, benchmarks)
#      tool-fc-link  linking Fortran objects with the C++ compiler (FLIBS)
#
#  Only targets that need a tool depend on its check.  Include this file
#  after the default goal of a makefile.  TOPDIR must be set.
#

PROBE = PROBE_CXX=$(call shell_quote,$(CXX) $(ULMBLAS_CXXFLAGS) $(CPPFLAGS) $(CXXFLAGS)) \
        PROBE_CC=$(call shell_quote,$(CC) $(CPPFLAGS) $(CFLAGS)) \
        PROBE_FC=$(call shell_quote,$(FC) $(FFLAGS)) \
        PROBE_LD=$(call shell_quote,$(LDFLAGS) $(LDLIBS)) \
        PROBE_FLIBS=$(call shell_quote,$(FLIBS)) \
        PROBE_AR=$(call shell_quote,$(AR) $(ARFLAGS)) \
        PROBE_RANLIB=$(call shell_quote,$(RANLIB)) \
        $(SHELL) $(TOPDIR)/mk/probe.sh

.PHONY : tool-cxx tool-ar tool-cc tool-fc tool-fc-link

tool-cxx :
	@$(PROBE) cxx

tool-ar : tool-cxx
	@$(PROBE) ar

tool-cc :
	@$(PROBE) cc

tool-fc :
	@$(PROBE) fc

tool-fc-link : tool-cxx tool-fc
	@$(PROBE) fc-link
