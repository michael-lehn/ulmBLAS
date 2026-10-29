#
#  cULMblas: GEMM with CUDA (independent of the rest of ulmBLAS).
#
#      make                 build build/test_gemm and build/bench_gemm
#      make check           run the correctness tests against cuBLAS
#      make check-memcheck  the quick tests under compute-sanitizer
#      make bench           run the benchmark against cuBLAS (BENCH_ARGS)
#      make plot            the plots of the steps from their saved results
#                           (gnuplot; independent of the CUDA build)
#      make clean           remove build/
#      make help            targets and settings
#
#  Settings (command line or environment):
#
#      CUDA_HOME    CUDA toolkit, if nvcc is not in PATH (default:
#                   /usr/local/cuda)
#      NVCC         the CUDA compiler                 (default: nvcc)
#      CUDA_ARCH    GPU architecture: native (the GPU of this machine), or
#                   e.g. sm_75, sm_80, sm_90; needed where nvcc sees no GPU
#                   (login nodes)                     (default: native)
#      CCBIN        host compiler for nvcc (-ccbin)   (default: nvcc's)
#      NVCCFLAGS    flags for nvcc                    (default: -O3)
#      TEST_ARGS    arguments of test_gemm (e.g. --quiet, --quick)
#      BENCH_ARGS   arguments of bench_gemm (e.g. --reps 20 --shapes ...)
#      MAGMA        MAGMA installation (with include/ and lib/): adds
#                   magma_dgemm and magmablas_dgemm as references to the
#                   tests and the benchmark                (default: none)
#      MAGMA_CFLAGS, MAGMA_LIBS  compile and link flags for MAGMA, if it is
#                   not installed as MAGMA/include, MAGMA/lib
#      GNUPLOT      gnuplot for make plot             (default: gnuplot)
#

CUDA_HOME  ?= /usr/local/cuda
NVCC       ?= $(shell command -v nvcc 2>/dev/null || \
                      { [ -x $(CUDA_HOME)/bin/nvcc ] && echo $(CUDA_HOME)/bin/nvcc; })
CUDA_ARCH  ?= native
NVCCFLAGS  ?= -O3
CCBIN      ?=
TEST_ARGS  ?=
BENCH_ARGS ?=
MAGMA      ?=
GNUPLOT    ?= gnuplot

ifneq ($(MAGMA),)
MAGMA_CFLAGS ?= -I$(MAGMA)/include
MAGMA_LIBS   ?= -L$(MAGMA)/lib -lmagma -Xlinker -rpath=$(MAGMA)/lib
endif
ifneq ($(MAGMA_CFLAGS)$(MAGMA_LIBS),)
WITH_MAGMA = -DCULMBLAS_WITH_MAGMA $(MAGMA_CFLAGS)
endif

BUILD      = build
ALL_FLAGS  = -std=c++17 -arch=$(CUDA_ARCH) $(if $(CCBIN),-ccbin $(CCBIN)) \
             -I. $(WITH_MAGMA) $(NVCCFLAGS)
LIBS       = $(MAGMA_LIBS) -lcublas

#  the steps of the tutorial
STEPS       = step01 step02 step03 step04 step05 step06 step07 step08 step09 step10 step11 step12 step13

LIB_SOURCES = culmblas/gemm.cu $(addsuffix /gemm.cu,$(STEPS))
LIB_OBJECTS = $(patsubst %.cu,$(BUILD)/%.o,$(LIB_SOURCES))
HEADERS     = culmblas/check.h culmblas/gemm.h support/support.h \
              support/magma.h $(addsuffix /gemm.h,$(STEPS))

PROGRAMS    = $(BUILD)/test_gemm $(BUILD)/bench_gemm

.PHONY : all check check-memcheck bench plot clean help require-nvcc require-gpu
.DELETE_ON_ERROR :

all : $(PROGRAMS)

require-nvcc :
	@if [ -z "$(NVCC)" ] || ! "$(NVCC)" --version > /dev/null 2>&1; then \
	    echo "error: the CUDA compiler nvcc was not found."; \
	    echo "  This project needs the CUDA Toolkit (nvcc, cuBLAS) and an NVIDIA GPU."; \
	    echo "  Put nvcc into PATH, or set CUDA_HOME=/path/to/cuda or NVCC=/path/to/nvcc."; \
	    exit 1; \
	fi

require-gpu :
	@if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then :; else \
	    echo "error: no NVIDIA GPU is usable on this machine (nvidia-smi -L fails)."; \
	    echo "  Build here with CUDA_ARCH=sm_XY if needed, but run the tests and"; \
	    echo "  the benchmark on a machine with an NVIDIA GPU."; \
	    exit 1; \
	fi

#  the flags are compiled into the benchmark, which prints them
$(BUILD)/flags : require-nvcc FORCE
	@mkdir -p $(BUILD)
	@echo '$(NVCC) $(ALL_FLAGS) $(LIBS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(BUILD)/%.o : %.cu $(HEADERS) $(BUILD)/flags
	@mkdir -p $(dir $@)
	$(NVCC) $(ALL_FLAGS) -c -o $@ $<

$(BUILD)/test_gemm : $(BUILD)/test/test_gemm.o $(LIB_OBJECTS)
	$(NVCC) $(ALL_FLAGS) -o $@ $^ $(LIBS)

$(BUILD)/bench/bench_gemm.o : bench/bench_gemm.cu $(HEADERS) $(BUILD)/flags
	@mkdir -p $(dir $@)
	$(NVCC) $(ALL_FLAGS) -DCULMBLAS_BUILD_FLAGS='"$(NVCC) $(ALL_FLAGS) $(LIBS)"' -c -o $@ $<

$(BUILD)/bench_gemm : $(BUILD)/bench/bench_gemm.o $(LIB_OBJECTS)
	$(NVCC) $(ALL_FLAGS) -o $@ $^ $(LIBS)

check : $(BUILD)/test_gemm require-gpu
	$(BUILD)/test_gemm $(TEST_ARGS)

check-memcheck : $(BUILD)/test_gemm require-gpu
	$(dir $(NVCC))compute-sanitizer --tool memcheck --error-exitcode 1 \
	    $(BUILD)/test_gemm --quick --quiet $(TEST_ARGS)

bench : $(BUILD)/bench_gemm require-gpu
	$(BUILD)/bench_gemm $(BENCH_ARGS)

plot :
	@for s in $(STEPS); do (cd $$s && $(GNUPLOT) plot.gp) || exit 1; done

clean :
	rm -rf $(BUILD)

help :
	@sed -n '2,/^$$/{s/^#  //;s/^#$$//;p;}' Makefile
	@echo 'Current settings:'
	@echo '  NVCC      = $(if $(NVCC),$(NVCC),(not found))'
	@echo '  CUDA_ARCH = $(CUDA_ARCH)'
	@echo '  CCBIN     = $(CCBIN)'
	@echo '  NVCCFLAGS = $(NVCCFLAGS)'
	@echo '  MAGMA     = $(if $(WITH_MAGMA),$(MAGMA_CFLAGS) $(MAGMA_LIBS),(not used))'

FORCE :
