#
#  The libraries of the interfaces, as rules generated for each build.
#
#  A build N is described by variables N.* and its rules are made by
#  $(eval $(call cblas_lib,N,CONFIG)) (C interface of BLAS (Basic Linear
#  Algebra Subprograms), interfaces/blas/C), f77blas_lib (Fortran 77
#  interface, interfaces/blas/F77), clapack_lib and f77lapack_lib (the C
#  and Fortran 77 interfaces of the LAPACK (Linear Algebra PACKage)
#  functions of ulmBLAS, interfaces/lapack/C and F77).  A build has
#  its own object directory, so the same sources are compiled side by side
#  with other flags: the test configurations, the shared library and its
#  interfaces with 64-bit integers, the backends of the dispatching
#  library.  Set before the call:
#
#      N.out       directory of the library and the generated headers
#      N.obj       directory of the objects
#      N.index     ULMBLAS_INDEX_TYPE of the interface
#      N.arch      ARCHFLAGS
#      N.config    CONFIG_CPPFLAGS (block sizes of a test configuration)
#      N.cxxflags  added to CXXFLAGS (-fPIC, macros of the shared library)
#      N.fflags    FFLAGS of the Fortran sources (f77blas_lib)
#      N.asm       yes: with the assembly kernels the flags select
#      N.tools     the checks of the tools (mk/tools.mk) done first
#      N.blas      the BLAS build whose header the LAPACK interface
#                  includes (clapack_lib, f77lapack_lib)
#
#  defaults: lib_defaults.  The build defines N.lib (the library), N.header
#  (its header), N.all (everything it builds) and N.objects.
#

PASTE := \#\#
COMMA := ,
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)

.PHONY : FORCE
FORCE :

#  upper case (the include guards of the generated headers)
uc = $(subst a,A,$(subst b,B,$(subst c,C,$(subst d,D,$(subst e,E,$(subst \
     f,F,$(subst g,G,$(subst h,H,$(subst i,I,$(subst j,J,$(subst k,K,$(subst \
     l,L,$(subst m,M,$(subst n,N,$(subst o,O,$(subst p,P,$(subst q,Q,$(subst \
     r,R,$(subst s,S,$(subst t,T,$(subst u,U,$(subst v,V,$(subst w,W,$(subst \
     x,X,$(subst y,Y,$(subst z,Z,$1))))))))))))))))))))))))))

#
#  The names of the configurations of the interfaces: library, header,
#  include guard, prefix and suffix of the functions.  C interface
#  (cfg.C.*: cblas, atl, cblas64) and Fortran 77 interface (cfg.F.*: f77,
#  shared, shared64, lbt); the LAPACK interfaces take the names of their
#  BLAS configuration.
#
cfg.cblas.prefix          = cblas_
cfg.cblas.uprefix         = ulmblas_
cfg.cblas.lib             = libcblas.a
cfg.cblas.header          = cblas.h
cfg.cblas.guard           = ULMBLAS_CBLAS_H
cfg.cblas.clapack.prefix  = cblas_
cfg.cblas.clapack.uprefix = ulmblas_
cfg.cblas.clapack.lib     = libclapack.a
cfg.cblas.clapack.header  = clapack.h
cfg.cblas.clapack.guard   = ULMBLAS_CLAPACK_H

cfg.atl.prefix            = cblas_
cfg.atl.uprefix           = ATL_
cfg.atl.lib               = libatlblas.a
cfg.atl.header            = atlblas.h
cfg.atl.guard             = ULMBLAS_ATLBLAS_H
cfg.atl.clapack.prefix    = cblas_
cfg.atl.clapack.uprefix   = ATL_
cfg.atl.clapack.lib       = libatllapack.a
cfg.atl.clapack.header    = atllapack.h
cfg.atl.clapack.guard     = ULMBLAS_ATLLAPACK_H

#  C interface with ILP64 (64-bit integer) indices and the suffix _64
#  (cblas_dgemm_64), the second interface of the shared library
#  (mk/shared.mk), as the reference CBLAS (C interface to the BLAS) built
#  with BUILD_INDEX64_EXT_API.
#  The ulmblas_ functions behind it get the suffix _ilp64 (not exported);
#  the globals of the reference CBLAS (RowMajorStrg, CBLAS_CallFromC) and
#  the helper transpose are those of the LP64 (32-bit integer) interface,
#  the thread functions and GEMM (general matrix-matrix product) with
#  arbitrary strides exist once (LP64).
cfg.cblas64.prefix        = cblas_
cfg.cblas64.suffix        = _64
cfg.cblas64.uprefix       = ulmblas_
cfg.cblas64.usuffix       = _ilp64
cfg.cblas64.lib           = libcblas64.a
cfg.cblas64.header        = cblas64.h
cfg.cblas64.guard         = ULMBLAS_CBLAS64_H
cfg.cblas64.exclude       = cblas_globals.cc transpose.cc threads.cc \
                            gemm_strided.cc

#  Fortran 77 interface: f77 (libf77blas.a), shared (the shared library,
#  32-bit integers, dgemm_), shared64 (its interface with 64-bit integers
#  and the suffix _64_, as the reference BLAS built with
#  BUILD_INDEX64_EXT_API and OpenBLAS with INTERFACE64=1 SYMBOLSUFFIX=64_),
#  lbt (libblastrampoline of Julia: 64-bit integers, suffix _64_, as in
#  OpenBLAS's ILP64 build)
cfg.f77.suffix            = _
cfg.shared.suffix         = _
cfg.shared64.suffix       = _64_
cfg.lbt.suffix            = _64_
$(foreach c,f77 shared shared64 lbt,\
    $(eval cfg.$c.lib = lib$cblas.a)\
    $(eval cfg.$c.header = $cblas.h)\
    $(eval cfg.$c.guard = ULMBLAS_$(call uc,$c)BLAS_H))

#
#  $(call lib_defaults,N): the settings a build did not set
#
lib_defaults = $(foreach v,index arch fflags,\
                   $(if $(filter undefined,$(origin $1.$v)),\
                        $(eval $1.$v = $$(lib.default.$v))))
lib.default.index  = $(ULMBLAS_INDEX_TYPE)
lib.default.arch   = $(ARCHFLAGS)
lib.default.fflags = $(FFLAGS)

#  The compile commands of build N (C++ and Fortran)
lib_cxxcompile = $(CXX) $($1.cppflags) \
                 -D'ULMBLAS_INDEX_TYPE_OF_BUILD=$($1.index)' \
                 $($1.config) $(CPPFLAGS) $(CXXSTD) $($1.arch) $(OPTFLAGS) \
                 $(SIMDFLAGS) $(OMPFLAGS) $(CXXFLAGS) $($1.cxxflags) \
                 $(DEPFLAGS) $(LIB_CXXFLAGS)
lib_fcompile   = $(FC) $(FOPTFLAGS) $($1.fflags)

#  the flags that select the micro kernels of build N
lib_kernelflags = $($1.arch) $($1.config) $(CPPFLAGS) $(CXXFLAGS) $($1.cxxflags)

#
#  $(eval $(call lib_rules,N)): objects and library of build N from N.src,
#  N.cxx (C++ sources), N.f (Fortran sources), N.headers (the generated
#  headers the C++ objects need) and the settings above.  The compile
#  commands are recorded in N.obj/compile.cmd (compile-f.cmd): a changed
#  command rebuilds the objects.
#
define lib_rules
$$(call lib_defaults,$1)
$1.cxx.o   := $$(patsubst %.cc,$$($1.obj)/%.o,$$($1.cxx))
$1.f.o     := $$(patsubst %.f,$$($1.obj)/%.o,$$($1.f))
$1.asm.o   := $$(if $$(filter yes,$$($1.asm)),$$(patsubst %.S,$$($1.obj)/%.o,\
                  $$(notdir $$(call asm_sources,$$(call lib_kernelflags,$1)))))
$1.objects := $$($1.cxx.o) $$($1.f.o) $$($1.asm.o)
lib.clean  += $$($1.obj) $$($1.lib) $$($1.header)

$$($1.lib) : $$($1.objects)
	@mkdir -p $$(@D)
	$$(call make_archive,$$@,$$($1.objects))

$$($1.obj)/compile.cmd : FORCE | $$($1.tools)
	@mkdir -p $$(@D)
	@$$(call record_command,$$(call lib_cxxcompile,$1),$$@)

$$($1.cxx.o) : $$($1.obj)/%.o : $$($1.src)/%.cc $$($1.headers) \
                                $$($1.obj)/compile.cmd
	$$(call lib_cxxcompile,$1) -c -o $$@ $$<

ifneq ($$(strip $$($1.asm.o)),)
$$($1.asm.o) : $$($1.obj)/%.o : $$(ULMBLAS_ASM_DIR)/%.S $$($1.obj)/compile.cmd
	$$(ASCOMPILE) -c -o $$@ $$<
endif

ifneq ($$(strip $$($1.f.o)),)
$$($1.obj)/compile-f.cmd : FORCE | $$($1.tools) tool-fc
	@mkdir -p $$(@D)
	@$$(call record_command,$$(call lib_fcompile,$1),$$@)

$$($1.f.o) : $$($1.obj)/%.o : $$($1.src)/%.f $$($1.obj)/compile-f.cmd
	$$(call lib_fcompile,$1) -c -o $$@ $$<
endif

lib.depfiles += $$($1.cxx.o:.o=.d) $$($1.asm.o:.o=.d)
endef

#
#  C interface of BLAS (CBLAS), configuration C: cblas, atl or cblas64
#
cblas_defs = -D'BLAS_HEADER=<$(cfg.$1.header)>' \
             -D'CBLAS(x)=$(cfg.$1.prefix) $(PASTE) x$(if $(cfg.$1.suffix), $(PASTE) $(cfg.$1.suffix))' \
             -D'ULMBLAS(x)=$(cfg.$1.uprefix) $(PASTE) x$(if $(cfg.$1.usuffix), $(PASTE) $(cfg.$1.usuffix))' \
             -D'CBLAS_H=$(cfg.$1.guard)' \
             -D'CBLAS_XERBLA_CALLBACK_T=cblas_xerbla$(cfg.$1.suffix)_callback_t' \
             -D'BLAS_FORTRAN_XERBLA_CALLBACK_T=blas_fortran_xerbla$(cfg.$1.suffix)_callback_t'

define cblas_lib
$1.src      := interfaces/blas/C
$1.cxx      := $$(filter-out $$(cfg.$2.exclude),\
                   $$(notdir $$(wildcard interfaces/blas/C/*.cc)))
$1.lib      := $$($1.out)/$$(cfg.$2.lib)
$1.header   := $$($1.out)/$$(cfg.$2.header)
$1.strided  := $$($1.out)/ulmblas_strided.h
$1.headers  := $$($1.header)
$1.defs     := $$(call cblas_defs,$2)
$1.cppflags  = -I$$($1.out) -I$$(TOPDIR) -DADD_ $$($1.defs)
$1.all      := $$($1.lib) $$($1.header) $$($1.strided)
$$(eval $$(call lib_rules,$1))
lib.clean   += $$($1.strided)

#  The header carries the index type (from the compile command): it is
#  regenerated when the command changes.
$$($1.header) : interfaces/blas/C/cblas.h.in $$($1.obj)/compile.cmd
	@mkdir -p $$(@D)
	$$(call generate_header,$$<,$$($1.defs) -D'ULMBLAS_INDEX_TYPE_OF_BUILD=$$($1.index)',$$@)

#  GEMM with arbitrary strides (gemm_strided.cc), not generated; one rule
#  per directory (cblas and atl share the top directory)
ifeq ($$(lib.strided.$$($1.strided)),)
lib.strided.$$($1.strided) := yes
$$($1.strided) : interfaces/blas/C/ulmblas_strided.h
	@mkdir -p $$(@D)
	cp $$< $$@
endif
endef

#
#  Fortran 77 interface of BLAS, configuration C: f77, shared, shared64 or
#  lbt.  Some functions are Fortran sources (N.f, by default all *.f; the
#  shared libraries take them from interfaces/blas/lbt/functions.c and set
#  N.f empty).
#
f77blas_defs = -D'BLAS_HEADER=<$(cfg.$1.header)>' \
               -D'F77BLAS(x)= x $(PASTE) $(cfg.$1.suffix)' \
               -D'F77BLAS_H=$(cfg.$1.guard)'

define f77blas_lib
$1.src      := interfaces/blas/F77
$1.cxx      := $$(notdir $$(wildcard interfaces/blas/F77/*.cc))
$$(if $$(filter undefined,$$(origin $1.f)),\
      $$(eval $1.f := $$(notdir $$(wildcard interfaces/blas/F77/*.f))))
$1.lib      := $$($1.out)/$$(cfg.$2.lib)
$1.header   := $$($1.out)/$$(cfg.$2.header)
$1.headers  := $$($1.header)
$1.defs     := $$(call f77blas_defs,$2)
$1.cppflags  = -I$$($1.out) -I$$(TOPDIR) $$($1.defs)
$1.all      := $$($1.lib) $$($1.header)
$1.config.name := $2
$$(eval $$(call lib_rules,$1))

$$($1.header) : interfaces/blas/F77/f77blas.h.in $$($1.obj)/compile.cmd
	@mkdir -p $$(@D)
	$$(call generate_header,$$<,$$($1.defs) -D'ULMBLAS_INDEX_TYPE_OF_BUILD=$$($1.index)',$$@)
endef

#
#  C interface of the LAPACK functions of ulmBLAS (?getrf, ?potrf, ...),
#  configuration C as for the C interface of BLAS (N.blas), whose header it
#  includes
#
clapack_defs = -D'BLAS_HEADER=<$(cfg.$1.header)>' \
               -D'LAPACK_HEADER=<$(cfg.$1.clapack.header)>' \
               -D'CLAPACK(x)=$(cfg.$1.clapack.prefix) $(PASTE) x' \
               -D'ULMLAPACK(x)=$(cfg.$1.clapack.uprefix) $(PASTE) x' \
               -D'CLAPACK_H=$(cfg.$1.clapack.guard)'

define clapack_lib
$1.src      := interfaces/lapack/C
$1.cxx      := $$(notdir $$(wildcard interfaces/lapack/C/*.cc))
$1.lib      := $$($1.out)/$$(cfg.$2.clapack.lib)
$1.header   := $$($1.out)/$$(cfg.$2.clapack.header)
$1.headers  := $$($$($1.blas).header) $$($1.header)
$1.defs     := $$(call clapack_defs,$2)
$1.cppflags  = -I$$($1.out) -I$$(TOPDIR) $$($1.defs)
$1.all      := $$($1.lib) $$($1.header)
$$(eval $$(call lib_rules,$1))

$$($1.header) : interfaces/lapack/C/clapack.h.in $$($$($1.blas).header) \
                $$($1.obj)/compile.cmd
	@mkdir -p $$(@D)
	$$(call generate_header,$$<,$$($1.defs),$$@)
endef

#
#  Fortran 77 interface of the LAPACK functions of ulmBLAS, configuration
#  C as for the Fortran 77 interface of BLAS (N.blas), whose header it
#  includes; its functions report invalid arguments with XERBLA of that
#  interface.  The include guard is named after N.guardname (default C).
#
f77lapack_defs = -D'BLAS_HEADER=<$(cfg.$1.header)>' \
                 -D'LAPACK_HEADER=<$(subst blas.h,lapack.h,$(cfg.$1.header))>' \
                 -D'F77BLAS(x)= x $(PASTE) $(cfg.$1.suffix)' \
                 -D'F77LAPACK_H=ULMBLAS_$(call uc,$2)LAPACK_H'

define f77lapack_lib
$1.src      := interfaces/lapack/F77
$1.cxx      := $$(notdir $$(wildcard interfaces/lapack/F77/*.cc))
$1.lib      := $$($1.out)/$$(subst blas.a,lapack.a,$$(cfg.$2.lib))
$1.header   := $$($1.out)/$$(subst blas.h,lapack.h,$$(cfg.$2.header))
$1.headers  := $$($$($1.blas).header) $$($1.header)
$1.defs     := $$(call f77lapack_defs,$2,$$(or $$($1.guardname),$2))
$1.cppflags  = -I$$($1.out) -I$$(TOPDIR) $$($1.defs)
$1.all      := $$($1.lib) $$($1.header)
$$(eval $$(call lib_rules,$1))

$$($1.header) : interfaces/lapack/F77/f77lapack.h.in $$($$($1.blas).header) \
                $$($1.obj)/compile.cmd
	@mkdir -p $$(@D)
	$$(call generate_header,$$<,$$($1.defs),$$@)
endef
