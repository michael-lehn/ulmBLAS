#
#  Shared library for LBT (libblastrampoline), the BLAS (Basic Linear
#  Algebra Subprograms) forwarding layer of Julia: the Fortran 77 interface
#  of ulmBLAS with ILP64 (64-bit integer) indices and the symbol suffix
#  _64_ (dgemm_64_), in one shared library
#
#      libulmblas_lbt.so      (Linux)
#      libulmblas_lbt.dylib   (macOS)
#      libulmblas_lbt.dll     (Windows, MinGW-w64)
#
#  The objects of the Fortran 77 interface are built with the
#  configuration lbt (mk/lib.mk) and -fPIC; the Fortran functions (DDOT,
#  ...) come from interfaces/blas/lbt/functions.c, so no Fortran compiler
#  is needed.  The library also contains GEMM (general matrix-matrix
#  product) with arbitrary strides (ulmblas_?gemm_strided,
#  interfaces/blas/C/ulmblas_strided.h); libblastrampoline does not know
#  these names and ignores them.  The
#  library links no C++ runtime (NO_CXX_RUNTIME in make.inc), so it does
#  not depend on the libstdc++ of the compiler (Julia brings its own,
#  possibly older one).  ARCHFLAGS select the micro kernels as for the
#  other libraries.  In Julia:
#
#      using LinearAlgebra
#      BLAS.lbt_forward("/path/libulmblas_lbt.so"; clear=false)
#
#  LAPACK=1: the library also contains ?getrf_64_ and ?potrf_64_ (the
#  Fortran 77 interface of the LAPACK (Linear Algebra PACKage) functions).
#  Off by default: libblastrampoline would then forward them to ulmBLAS
#  instead of OpenBLAS, whose versions are multithreaded; ulmBLAS factors
#  in parallel only if it is built with OpenMP (Open Multi-Processing).
#
LAPACK ?= 0

lbt.dir := interfaces/blas/lbt/build.lbt
lbt.out := .

lbt.f77.out      := $(lbt.dir)
lbt.f77.obj      := $(lbt.dir)/obj
lbt.f77.f        :=
lbt.f77.index    := int64_t
lbt.f77.cxxflags := -fPIC
lbt.f77.asm      := yes
lbt.f77.tools    := tool-ar
$(eval $(call f77blas_lib,lbt.f77,lbt))

lbt.lapack.out      := $(lbt.dir)
lbt.lapack.obj      := $(lbt.dir)/obj-lapack
lbt.lapack.index    := int64_t
lbt.lapack.cxxflags := -fPIC
lbt.lapack.blas     := lbt.f77
lbt.lapack.tools    := tool-ar
$(eval $(call f77lapack_lib,lbt.lapack,lbt))

lbt.libs    = $(lbt.f77.lib) $(if $(filter 1,$(LAPACK)),$(lbt.lapack.lib))
lbt.objects = $(lbt.dir)/functions.o $(lbt.dir)/gemm_strided.o

ifeq ($(shared.platform),mingw)
#  Windows (MinGW-w64): all global symbols of the objects are exported (no
#  symbol is marked dllexport), winpthread is linked statically, so the
#  DLL (dynamic-link library) only depends on DLLs of the system.
lbt.lib   = $(lbt.out)/libulmblas_lbt.dll
lbt.link  = -shared -Wl,--whole-archive $(lbt.libs) -Wl,--no-whole-archive \
            -static $(or $(NOSTDLIBXX),-static-libstdc++) -static-libgcc \
            -Wl,--no-undefined
else ifeq ($(shared.platform),darwin)
lbt.lib   = $(lbt.out)/libulmblas_lbt.dylib
lbt.link  = -dynamiclib -install_name @rpath/libulmblas_lbt.dylib \
            $(foreach l,$(lbt.libs),-Wl$(COMMA)-force_load$(COMMA)$(l)) \
            $(NOSTDLIBXX)
else
lbt.lib   = $(lbt.out)/libulmblas_lbt.so
lbt.link  = -shared -Wl,-soname,libulmblas_lbt.so \
            -Wl,--whole-archive $(lbt.libs) -Wl,--no-whole-archive \
            $(or $(NOSTDLIBXX),-static-libstdc++) -static-libgcc \
            -Wl,--no-undefined
endif
lib.clean += $(lbt.dir) $(lbt.lib)

$(lbt.objects) : | tool-ar

#  the compile commands of the objects (a changed one rebuilds them)
lbt.ccmd := $(lbt.dir)/compile-c.cmd
$(lbt.ccmd) : FORCE | tool-ar
	@mkdir -p $(@D)
	@$(call record_command,$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) ; $(CXX) $(ULMBLAS_CXXFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LIB_CXXFLAGS),$@)

$(lbt.dir)/functions.o : interfaces/blas/lbt/functions.c $(lbt.f77.header) \
                         $(lbt.ccmd)
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC -I$(lbt.dir) \
	      -D'BLAS_HEADER=<lbtblas.h>' -D'F77BLAS(x)=x$(PASTE)_64_' \
	      -c -o $@ $<

#  GEMM with arbitrary strides, from the C interface
$(lbt.dir)/gemm_strided.o : interfaces/blas/C/gemm_strided.cc \
                            interfaces/blas/C/ulmblas_strided.h \
                            $(lbt.ccmd) | $(lbt.f77.header)
	$(CXX) -I$(TOPDIR) -D'ULMBLAS(x)=ulmblas_$(PASTE)x' $(ULMBLAS_CXXFLAGS) \
	       $(CPPFLAGS) $(CXXFLAGS) $(LIB_CXXFLAGS) -fPIC -c -o $@ $<

#  The objects must not refer to the C++ runtime (NO_CXX_RUNTIME in
#  make.inc; this also excludes functions that throw: no C++ exception may
#  pass through the interface, Julia calls it from C).  Checked before
#  linking.
$(lbt.lib) : $(lbt.libs) $(lbt.objects)
	@$(NO_CXX_RUNTIME) $^
	@mkdir -p $(@D)
	$(CXX) $(ARCHFLAGS) $(LDFLAGS) -o $@ $(lbt.objects) $(lbt.link) \
	       $(OMPLIBS) -lm
