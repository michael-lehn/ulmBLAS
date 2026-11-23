#
#  Settings of the build of ulmBLAS (read by GNUmakefile; the whole build is
#  one make, no make calls another, see mk/README.md).
#
#  ulmBLAS is built with GNU Make 3.81 or newer.  On macOS /usr/bin/make is
#  GNU Make 3.81, a newer GNU Make from Homebrew or MacPorts is usually
#  installed as 'gmake'.  On the BSDs GNU Make is installed as 'gmake'.
#
#  Every setting below can be overridden on the command line or through the
#  environment, e.g.
#
#      make CXX=clang++ CXXFLAGS=-g check
#      make OPTFLAGS=-O0 CXXFLAGS=-g          (debug build without optimization)
#
#  Variables meant to be set by users:
#
#      CXX        C++ compiler (C++11 or newer)               default: c++
#      CC         C compiler (Netlib CBLAS test drivers, bench) default: cc
#      FC         Fortran compiler (F77 interface, refblas,     default: gfortran
#                 Netlib test drivers)
#      AR, ARFLAGS, RANLIB   archiver                         default: ar cr, ranlib
#      CXXSTD     C++ language standard                       default: -std=c++11
#      ARCHFLAGS  architecture flags                          default: see below
#      OPTFLAGS   C++ optimization       default: -O3 -fomit-frame-pointer
#      FOPTFLAGS  Fortran optimization                        default: -O3
#      COPTFLAGS  C optimization                              default: -O2
#      CPPFLAGS, CXXFLAGS, CFLAGS, FFLAGS, LDFLAGS, LDLIBS
#      FLIBS      libraries needed to link Fortran objects with the C++
#                 compiler                                    default: see below
#      ULMBLAS_INDEX_TYPE
#                 integer type of dimensions, increments, leading
#                 dimensions and indices in the C, Fortran 77 and LAPACK
#                 interfaces (e.g. long, int64_t)             default: int
#
#  Flags required by the build itself (include paths, macros selecting the
#  interface variant, test block sizes) are kept in separate variables, so
#  overriding CXXFLAGS etc. never breaks the build.
#
#  Precedence (GNU Make): a variable given on the command line overrides the
#  assignments in the makefiles; a variable
#  from the environment is only used by the '?=' assignments below, which
#  do not assign a variable that is already defined.  The optimization flags
#  are kept in OPTFLAGS, FOPTFLAGS and COPTFLAGS, which come before CXXFLAGS,
#  FFLAGS and CFLAGS on every command line: CXXFLAGS from the environment
#  (often set for include paths) does not remove the default optimization,
#  and an optimization flag in CXXFLAGS (e.g. -O0) takes precedence because
#  for GCC and clang the last -O option wins.
#

#
#  GNU Make version check
#
ULMBLAS_MAKE_MIN := 3.81

ULMBLAS_MAKE_MAJOR := $(word 1,$(subst ., ,$(MAKE_VERSION)))
ULMBLAS_MAKE_MINOR := $(word 2,$(subst ., ,$(MAKE_VERSION)))

#  Too old: major version < 3, or 3 and minor version < 81 (the minor
#  versions of GNU Make 3 have two digits, so they compare as strings).  Only
#  functions known to old versions of GNU Make are used here.
ULMBLAS_MAKE_TOO_OLD :=
ifeq ($(MAKE_VERSION),)
ULMBLAS_MAKE_TOO_OLD := yes
endif
ifneq ($(filter 0 1 2,$(ULMBLAS_MAKE_MAJOR)),)
ULMBLAS_MAKE_TOO_OLD := yes
endif
ifeq ($(ULMBLAS_MAKE_MAJOR),3)
ifneq ($(firstword $(sort $(ULMBLAS_MAKE_MINOR) 81)),81)
ULMBLAS_MAKE_TOO_OLD := yes
endif
endif

ifneq ($(ULMBLAS_MAKE_TOO_OLD),)
$(error GNU Make $(MAKE_VERSION) is too old: ulmBLAS requires GNU Make $(ULMBLAS_MAKE_MIN) or newer. If a newer GNU Make is installed as 'gmake', run 'gmake' instead of '$(MAKE)')
endif

#
#  Compilers and tools.  GNU Make's built-in default for CXX differs between
#  versions and platforms (g++, c++, clang++), hence c++, the system's C++
#  compiler, is used unless CXX is set explicitly.  The built-in default
#  FC=f77 is replaced by gfortran.
#
ifeq ($(origin CXX),default)
CXX := c++
endif

ifeq ($(origin FC),default)
FC := gfortran
endif

ifeq ($(origin ARFLAGS),default)
ARFLAGS := cr
endif

RANLIB  ?= ranlib

#
#  Compiler flags.  The defaults of the C++ and Fortran flags are the flags
#  ulmBLAS has always been built with.  C is only used for the Netlib CBLAS
#  test drivers and the benchmark programs.
#
CXXSTD    ?= -std=c++11
OPTFLAGS  ?= -O3 -fomit-frame-pointer
FOPTFLAGS ?= -O3
COPTFLAGS ?= -O2
CXXFLAGS  ?= -Wall
FFLAGS    ?= -Wall -fimplicit-none

#
#  ARCHFLAGS: on x86 the optimized SSE kernels require SSE3.  The flags are
#  only used if the C++ compiler accepts them for its target (e.g. not on
#  ARM, where the portable reference kernels are used).  Exported for the
#  scripts the recipes run.
#
ULMBLAS_ARCHFLAGS_DEFAULT = $(shell echo | $(CXX) -msse3 -mfpmath=sse -x c++ -E - \
                                  >/dev/null 2>&1 && echo -msse3 -mfpmath=sse)
ifeq ($(origin ARCHFLAGS),undefined)
ARCHFLAGS := $(ULMBLAS_ARCHFLAGS_DEFAULT)
endif
export ARCHFLAGS

#
#  TARGET_MACHINE: the platform of the compiler (cross compilers work, e.g.
#  CXX=x86_64-w64-mingw32-g++); it selects the shared libraries
#  (mk/shared.mk, mk/dispatch.mk, mk/install.mk).  Computed when needed,
#  and only once.
#
TARGET_MACHINE = $(eval TARGET_MACHINE := $(shell $(CXX) -dumpmachine))$(TARGET_MACHINE)

#
#  FLIBS: libraries for linking Fortran objects with the C++ compiler as
#  linker front end (the C++ runtime then is the one of CXX).  For gfortran
#  the directory of libgfortran is determined by asking gfortran.  For other
#  Fortran compilers set FLIBS explicitly.  The value is computed only when
#  needed, and only once.
#
ifeq ($(origin FLIBS),undefined)
FLIBS = $(eval FLIBS := $(ULMBLAS_FLIBS))$(FLIBS)
endif

ULMBLAS_FLIBS = $(shell \
    for lib in libgfortran.dylib libgfortran.so libgfortran.a; do         \
        path=`$(FC) -print-file-name=$$lib 2>/dev/null`;                  \
        if [ -n "$$path" ] && [ "$$path" != "$$lib" ]; then               \
            echo "-L`dirname $$path` -lgfortran"; break;                  \
        fi;                                                               \
    done) -lm

#
#  ULMBLAS_INDEX_TYPE: also taken from CPPFLAGS=-DULMBLAS_INDEX_TYPE=...
#  The generated headers define the macro ULMBLAS_INDEX_TYPE with this
#  type, the interface sources get it from these headers.  The type is
#  also on every compile command (ULMBLAS_INDEX_TYPE_OF_BUILD), so a
#  changed type regenerates the headers and rebuilds the objects.
#
ifeq ($(origin ULMBLAS_INDEX_TYPE),undefined)
ULMBLAS_INDEX_TYPE := $(or $(patsubst -DULMBLAS_INDEX_TYPE=%,%,\
                               $(filter -DULMBLAS_INDEX_TYPE=%,$(CPPFLAGS))),int)
endif
ULMBLAS_INDEX_DEFS = -D'ULMBLAS_INDEX_TYPE_OF_BUILD=$(ULMBLAS_INDEX_TYPE)'

#
#  OPENMP=1: the LAPACK functions (?getrf, ?potrf) factor larger matrices
#  with OpenMP tasks (OpenMP 5.0: GCC 9 or newer, clang 12 or newer); off
#  by default.  OMPFLAGS are added to the compile flags, OMPLIBS are needed
#  to link programs and shared libraries.  macOS (Apple clang): libomp of
#  Homebrew by default.  At run time the task priorities need the
#  environment variable OMP_MAX_TASK_PRIORITY (e.g. 3).
#
OPENMP ?= 0
ifeq ($(OPENMP),1)
ifeq ($(shell uname -s),Darwin)
OMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
OMPFLAGS   ?= -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
OMPLIBS    ?= -L$(OMP_PREFIX)/lib -lomp
else
OMPFLAGS   ?= -fopenmp
OMPLIBS    ?= -fopenmp
endif
else
OMPFLAGS   =
OMPLIBS    =
endif

#
#  SIMDFLAGS: the SIMD directives of OpenMP (#pragma omp simd), for the
#  vectorized sums of level 1 (ulmblas/level1/kernel/reduce.h); no OpenMP
#  runtime, independent of OPENMP.  SIMDFLAGS= turns them off (the sums are
#  then computed in order).
#
SIMDFLAGS ?= -fopenmp-simd

#
#  Flags that are always used.  The compile commands of the builds are made
#  in mk/lib.mk: each build adds its include paths and interface macros, a
#  test configuration its block sizes (CONFIG_CPPFLAGS), a backend of the
#  dispatching library its ARCHFLAGS.
#
DEPFLAGS          = -MMD -MP
ULMBLAS_CXXFLAGS  = $(CXXSTD) $(ARCHFLAGS) $(OPTFLAGS) $(SIMDFLAGS) $(OMPFLAGS)
ULMBLAS_CFLAGS    = $(COPTFLAGS)

FCOMPILE   = $(FC) $(FOPTFLAGS) $(FFLAGS)

#
#  The objects of the libraries are compiled without exceptions: ulmBLAS
#  throws none, and none may pass through the C and Fortran interfaces.
#  Without them the objects need no exception handling of the C++ runtime
#  (personality routine, _Unwind_Resume, std::terminate): the libraries
#  link no C++ runtime (checked by NO_CXX_RUNTIME before linking).
#
LIB_CXXFLAGS  = -fno-exceptions

#
#  NO_CXX_RUNTIME: the shared libraries link no C++ runtime (libstdc++,
#  libc++, libsupc++, libc++abi): C++ is only the language of the
#  implementation, the interfaces are C and Fortran.  Before linking, the
#  objects are checked for references to functions of the runtime (symbols
#  of std:: that are not inline, operator new and delete, exception
#  handling, guards of static variables, __cxa_thread_atexit, ...); the
#  ones of the C library (__cxa_atexit, __cxa_finalize) are allowed.
#  NO_CXX_RUNTIME is the check, a shell command; the objects are its
#  arguments.
#
#  NOSTDLIBXX: the flag that links without the C++ standard library,
#  -nostdlib++ (clang, GCC 13 or newer); empty if the compiler does not
#  know it (then the libraries are linked with -static-libstdc++, which
#  adds nothing: no object refers to it).
#
CXX_RUNTIME_SYMBOLS = '^_?_Z(N?K?St|T[VIS]N?St|n[wa]|d[la]|T[VIS]N10__cxxabiv)|^_?__cxa_|^_?__gxx_|^_?_Unwind_'
CXX_RUNTIME_ALLOWED = '^_?__cxa_(atexit|finalize)$$'
NO_CXX_RUNTIME = check() { \
	    if $(NM) -u "$$@" | awk '{ print $$NF }' \
	       | grep -E $(CXX_RUNTIME_SYMBOLS) | grep -vE $(CXX_RUNTIME_ALLOWED); \
	    then \
		echo "error: objects refer to the C++ runtime (above)"; \
		return 1; \
	    fi; \
	}; check
NM ?= nm
NOSTDLIBXX := $(shell $(CXX) -nostdlib++ -E -x c++ /dev/null -o /dev/null \
                >/dev/null 2>&1 && echo -nostdlib++)

#
#  Micro kernels that are assembly files (ulmblas/level3/ukernel/openblas/
#  asm/*.S) are assembled into a library if they are selected, i.e. if the
#  flags of its build (ARCHFLAGS, CPPFLAGS, CXXFLAGS or the CONFIG_CPPFLAGS
#  of a test configuration) define their selection macro, or if the build
#  has AVX2 (x86-64: every vendored kernel of the instruction set is
#  compiled in, ulmblas/level3/ukernel/kernels.h, also the assembly
#  kernels).  $(call asm_sources,FLAGS): the assembly sources of a build
#  with these flags.  They are built with ASCOMPILE (the C++ compiler
#  driver preprocesses *.S files and runs the assembler).
#
ULMBLAS_ASM_DIR = $(TOPDIR)/ulmblas/level3/ukernel/openblas/asm
asm_selected    = $(filter -DULMBLAS_DGEMM_UKERNEL_openblas_$2 \
                           -DULMBLAS_DGEMM_UKERNEL_openblas_$2=%,$1)
asm_avx2        = $(if $(filter x86_64-% amd64-%,$(TARGET_MACHINE)),$(shell \
                      echo | $(CXX) $1 -dM -E -x c++ - 2>/dev/null \
                      | grep -q '__AVX2__ 1' && echo yes))
asm_sources     = $(if $(or $(call asm_selected,$1,haswell_4x8), \
                            $(call asm_avx2,$1)), \
                       $(ULMBLAS_ASM_DIR)/haswell_4x8.S)

ASCOMPILE  = $(CXX) -I$(TOPDIR) $(CPPFLAGS) $(DEPFLAGS)

#
#  Generate a header from a template (*.h.in) with the C++ preprocessor.
#  -P suppresses line markers, -CC keeps comments.  The template undefines
#  __cplusplus on purpose, hence -w.
#
#      $(call generate_header,template,macros,output)
#
generate_header = $(CXX) -x c++ -E -P -CC -w $2 $1 > $3.tmp \
                  && sed 's,^ *\#,\#,' $3.tmp > $3 && rm -f $3.tmp \
                  || { rm -f $3.tmp; exit 1; }

#
#  Record a command line in a file that is only rewritten if the command line
#  changed.  Objects depending on this file are rebuilt after the compiler or
#  the flags have changed.  Use with a FORCE prerequisite:
#
#      $(BUILD_DIR)/compile.cmd : FORCE | $(BUILD_DIR)
#              @$(call record_command,$(CXXCOMPILE),$@)
#
shell_quote    = '$(subst ','\'',$1)'
record_command = printf '%s\n' $(call shell_quote,$(strip $1)) > $2.tmp$$$$; \
                 if cmp -s $2.tmp$$$$ $2; then rm -f $2.tmp$$$$; \
                 else mv -f $2.tmp$$$$ $2; fi

#
#  Create a static library from object files.
#
#      $(call make_archive,library,objects)
#
make_archive = rm -f $1 && $(AR) $(ARFLAGS) $1 $2 && $(RANLIB) $1

#
#  Flags used for building the ATLAS benchmark suite
#
BENCH_CFLAGS := -c -DL2SIZE=4194304 -DAdd_ -DF77_INTEGER=int -DStringSunStyle \
                -DATL_SSE2 -DDREAL
