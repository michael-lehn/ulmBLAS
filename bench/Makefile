TOPDIR = ..
include $(TOPDIR)/make.inc

BLAS_VARIANTS = $(patsubst config/make.%.inc, %, $(wildcard config/make.*.inc))
BENCH_CASES   = $(patsubst cases/make.%.inc, %, $(wildcard cases/make.*.inc))

USE_BLAS_VARIANTS ?= $(BLAS_VARIANTS)
USE_BENCH_CASES   ?= $(BENCH_CASES)

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
CFLAGS              = $(BENCH_CFLAGS)

LIBTSTATLAS = libtstatlas.a

all : $(BENCHMARK_SUMMARIES) $(BENCHMARK_PLOTS)

benchmark-suite : $(USE_BLAS_VARIANTS)

%.plot : $(BENCHMARK_SUMMARIES)
	make -C . -f Makefile.plot CASE=${*F} DIR=${@D} BLAS_VARIANTS="$(USE_BLAS_VARIANTS)"

$(LIBTSTATLAS) : $(ATL_OBJECTFILES)
	ar r $(LIBTSTATLAS) $(ATL_OBJECTFILES)
	ranlib $(LIBTSTATLAS)

$(BLAS_VARIANTS) : $(LIBTSTATLAS)
	make -C . -f Makefile.config CONFIG=$@ BENCH_DIR_PREFIX=$(BENCH_DIR_PREFIX)

$(BENCHMARK_DIRS) :
	mkdir -p $(BENCHMARK_DIRS)

$(BENCHMARK_TOPDIR)/%.summ : $(BENCHMARK_DIRS) $(USE_BLAS_VARIANTS)
	make -C . -f Makefile.bench CASE=$(*D) CONFIG=$(*F) DIR=${@D}

clean :
	rm -rf $(BENCH_DIR_PREFIX).*
	rm -f $(ATL_OBJECTFILES) $(LIBTSTATLAS)
	rm -rf $(BENCHMARK_TOPDIR)
