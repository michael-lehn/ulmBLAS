#
#  Task-based LAPACK routines of ulmBLAS (dgetrf, dpotrf) with OpenMP.
#
#      make                    library build/libulmlapack.a, test, benchmarks
#      make check              correctness tests (quick set)
#      make check-full         all parameter combinations
#
#  Settings (command line or environment):
#
#      CXX        C++ compiler with OpenMP 5.0 (depend iterators): GCC >= 9,
#                 clang >= 12                               default: c++
#      ARCHFLAGS  selects the ulmBLAS micro kernels, as for the library,
#                 e.g. on Ice Lake (AVX-512, OpenBLAS's SKYLAKEX kernel):
#                 -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12
#      OMPFLAGS   compile flags for OpenMP          default: -fopenmp
#                 (Apple clang with Homebrew libomp: see below)
#      OMPLIBS    link flags for OpenMP             default: -fopenmp
#      REFLIBS    reference LAPACK (tests) and the library compared in the
#                 benchmarks: OpenBLAS              default: -lopenblas
#
#  On macOS with Apple clang and Homebrew (libomp, openblas) the defaults
#  below are used automatically.
#
TOPDIR    = ../..
CXX      ?= c++
CXXSTD   ?= -std=c++11
OPTFLAGS ?= -O3 -fomit-frame-pointer
ARCHFLAGS ?=
CXXFLAGS ?= -Wall

ifeq ($(shell uname -s),Darwin)
BREW     ?= /opt/homebrew
OMPFLAGS ?= -Xpreprocessor -fopenmp -I$(BREW)/opt/libomp/include
OMPLIBS  ?= -L$(BREW)/opt/libomp/lib -lomp
REFLIBS  ?= -L$(BREW)/opt/openblas/lib -lopenblas
else
OMPFLAGS ?= -fopenmp
OMPLIBS  ?= -fopenmp
REFLIBS  ?= -lopenblas
endif

BUILD    = build
CPPFLAGS_ALL = -I$(TOPDIR) -Iinclude -Isrc
COMPILE  = $(CXX) $(CXXSTD) $(OPTFLAGS) $(ARCHFLAGS) $(CPPFLAGS_ALL) \
           $(CPPFLAGS) $(OMPFLAGS) $(CXXFLAGS) -MMD -MP

LIB      = $(BUILD)/libulmlapack.a
LIBOBJS  = $(BUILD)/getrf.o $(BUILD)/potrf.o

.PHONY : all lib bench-core check check-full check-bounds check-neighbor check-deps \
         check-determinism clean

all : lib $(BUILD)/test $(BUILD)/bench-ulmblas $(BUILD)/bench-openblas \
      $(BUILD)/bench-core-ulmblas $(BUILD)/bench-core-openblas

lib : $(LIB)

$(BUILD) :
	mkdir -p $@

$(BUILD)/%.o : src/%.cc | $(BUILD)
	$(COMPILE) -c $< -o $@

$(LIB) : $(LIBOBJS)
	rm -f $@ && ar cr $@ $^ && ranlib $@

$(BUILD)/test.o : test/test.cc | $(BUILD)
	$(COMPILE) -c $< -o $@

$(BUILD)/test : $(BUILD)/test.o $(LIB)
	$(CXX) $(OPTFLAGS) $^ $(REFLIBS) $(OMPLIBS) -o $@

#  The benchmark program is compiled twice: once calling ulmlapack, once
#  calling the LAPACK routines of the reference library (OpenBLAS).  Both
#  are linked with the reference library (setup and checks).
$(BUILD)/bench-ulmblas.o : bench/bench.cc | $(BUILD)
	$(COMPILE) -DBENCH_ULMBLAS -c $< -o $@

$(BUILD)/bench-openblas.o : bench/bench.cc | $(BUILD)
	$(COMPILE) -DBENCH_OPENBLAS -c $< -o $@

$(BUILD)/bench-ulmblas : $(BUILD)/bench-ulmblas.o $(LIB)
	$(CXX) $(OPTFLAGS) $^ $(REFLIBS) $(OMPLIBS) -o $@

$(BUILD)/bench-openblas : $(BUILD)/bench-openblas.o
	$(CXX) $(OPTFLAGS) $^ $(REFLIBS) $(OMPLIBS) -o $@

#  ?getrf, ?potrf for s, d, c, z: the templates in ulmblas/lapack (with
#  OpenMP) and OpenBLAS (bench/core.cc)
$(BUILD)/bench-core-ulmblas : bench/core.cc | $(BUILD)
	$(COMPILE) -DBENCH_ULMBLAS $< $(REFLIBS) $(OMPLIBS) -o $@

$(BUILD)/bench-core-openblas : bench/core.cc | $(BUILD)
	$(COMPILE) -DBENCH_OPENBLAS $< $(REFLIBS) $(OMPLIBS) -o $@

#  oneMKL (mkl_rt, its default Intel threading; MKLROOT from the module)
$(BUILD)/bench-core-mkl : bench/core.cc | $(BUILD)
	$(CXX) $(CXXSTD) $(OPTFLAGS) -DBENCH_MKL $< -L$(MKLROOT)/lib -lmkl_rt \
	       -Wl,-rpath,$(MKLROOT)/lib -o $@

bench-core : $(BUILD)/bench-core-ulmblas $(BUILD)/bench-core-openblas

check : $(BUILD)/test
	$(BUILD)/test quick

#  The ulmBLAS kernels never write outside their output operand, for every
#  call shape of the task graphs (no OpenMP, no reference library).
$(BUILD)/bounds : test/bounds.cc src/kernels.h | $(BUILD)
	$(CXX) -std=c++14 $(OPTFLAGS) $(ARCHFLAGS) $(CPPFLAGS_ALL) $(CPPFLAGS) $< -o $@

check-bounds : $(BUILD)/bounds
	$(BUILD)/bounds

#  No kernel loads and stores back an element next to its 8 x 8 output tile
#  (a lost update in parallel); a second thread writes those elements.
$(BUILD)/neighbor : test/neighbor.cc src/kernels.h | $(BUILD)
	$(CXX) -std=c++14 $(OPTFLAGS) $(ARCHFLAGS) $(CPPFLAGS_ALL) $(CPPFLAGS) $< -pthread -o $@

check-neighbor : $(BUILD)/neighbor
	$(BUILD)/neighbor 100000

#  The test suite with the dependency checker (src/checkdeps.h): aborts if
#  two conflicting accesses to a tile ran out of order or overlapped.
check-deps :
	$(MAKE) BUILD=$(BUILD)-checkdeps CPPFLAGS="$(CPPFLAGS) -DULMLAPACK_CHECK_DEPS" \
	        $(BUILD)-checkdeps/test
	$(BUILD)-checkdeps/test

#  Bitwise reproducibility for fixed parameters (DETERMINISM_SECONDS).
DETERMINISM_SECONDS ?= 60
$(BUILD)/determinism : test/determinism.cc $(LIB)
	$(CXX) $(CXXSTD) $(OPTFLAGS) $(CPPFLAGS_ALL) $(OMPFLAGS) $^ $(REFLIBS) $(OMPLIBS) -o $@

check-determinism : $(BUILD)/determinism
	$(BUILD)/determinism $(DETERMINISM_SECONDS)

check-full : $(BUILD)/test
	$(BUILD)/test

clean :
	rm -rf $(BUILD)

-include $(wildcard $(BUILD)/*.d)
