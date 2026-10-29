#
#  Benchmarks (ATLAS test and timing programs, long running).  Usually
#  invoked from the top-level directory with 'make bench'.
#
#  USE_BLAS_VARIANTS  BLAS libraries compared (config/make.*.inc).  Default:
#                     ulmBLAS and netlib (Netlib reference BLAS).  The
#                     variants atlas and mkl need ATLAS_LIBDIR and MKLROOT.
#  USE_BENCH_CASES    benchmark cases (cases/make.*.inc), default: all
#
#  Results: benchmarks/<case>/<variant>.summ, plots (if gnuplot is found)
#  benchmarks/<case>/*.svg
#
TOPDIR = ..
include $(TOPDIR)/make.inc

BLAS_VARIANTS = $(patsubst config/make.%.inc, %, $(wildcard config/make.*.inc))
BENCH_CASES   = $(patsubst cases/make.%.inc, %, \
                           $(filter-out cases/make.inc, \
                                        $(wildcard cases/make.*.inc)))

USE_BLAS_VARIANTS ?= ulmBLAS netlib
USE_BENCH_CASES   ?= $(BENCH_CASES)

GNUPLOT           ?= gnuplot
HAVE_GNUPLOT      := $(shell command -v $(GNUPLOT) >/dev/null 2>&1 && echo yes)

BENCH_DIR_PREFIX    = bench

BENCHMARK_TOPDIR    = benchmarks
BENCHMARK_DIRS      = $(patsubst %,$(BENCHMARK_TOPDIR)/%,$(USE_BENCH_CASES))
BENCHMARK_SUMMARIES = $(foreach case, $(USE_BENCH_CASES),          \
                           $(foreach blas, $(USE_BLAS_VARIANTS),   \
                                $(BENCHMARK_TOPDIR)/$(case)/$(blas).summ))
BENCHMARK_PLOTS     = $(foreach case, $(USE_BENCH_CASES),          \
                           $(BENCHMARK_TOPDIR)/$(case).plot)

ATL_OBJECTFILES     = $(patsubst %.c, %.o, $(wildcard ATL_*.c))
ATL_OBJECTFILES    += $(patsubst %.f, %.o, $(wildcard ATL_*.f))
LIBTSTATLAS = libtstatlas.a

.PHONY : all plots benchmark-suite clean $(BLAS_VARIANTS)

ifeq ($(HAVE_GNUPLOT),yes)
all : $(BENCHMARK_SUMMARIES) $(BENCHMARK_PLOTS)
else
all : $(BENCHMARK_SUMMARIES)
	@echo "note: '$(GNUPLOT)' not found, no plots created (set GNUPLOT=...)."
	@echo "      Results: $(BENCHMARK_SUMMARIES)"
endif

plots : $(BENCHMARK_PLOTS)

benchmark-suite : $(USE_BLAS_VARIANTS)

%.plot : $(BENCHMARK_SUMMARIES)
	@command -v $(GNUPLOT) >/dev/null 2>&1 \
	 || { echo "error: '$(GNUPLOT)' not found, needed for plots" >&2; exit 1; }
	$(MAKE) -f Makefile.plot CASE=${*F} DIR=${@D} BLAS_VARIANTS="$(USE_BLAS_VARIANTS)"

%.o : %.c
	$(CC) $(BENCH_CFLAGS) $(ULMBLAS_CFLAGS) $(CFLAGS) -o $@ $<

$(LIBTSTATLAS) : $(ATL_OBJECTFILES)
	$(call make_archive,$@,$(ATL_OBJECTFILES))

$(BLAS_VARIANTS) : $(LIBTSTATLAS)
	$(MAKE) -f Makefile.config CONFIG=$@ BENCH_DIR_PREFIX=$(BENCH_DIR_PREFIX)

$(BENCHMARK_DIRS) :
	mkdir -p $(BENCHMARK_DIRS)

$(BENCHMARK_TOPDIR)/%.summ : $(BENCHMARK_DIRS) $(USE_BLAS_VARIANTS)
	$(MAKE) -f Makefile.bench CASE=$(notdir $(*D)) CONFIG=$(*F) DIR=${@D}

clean :
	rm -rf $(BENCH_DIR_PREFIX).*
	rm -f $(ATL_OBJECTFILES) $(LIBTSTATLAS)
	rm -rf $(BENCHMARK_TOPDIR)
