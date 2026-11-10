#
#  Dispatching shared library for macOS on Apple Silicon (see dispatch.c):
#
#      libulmblas.dylib        selects a backend at load time
#      libulmblas_neon.dylib   NEON kernels, every Apple Silicon CPU
#      libulmblas_sme.dylib    SME2 kernels, Apple M4 and later
#
#  The backends are the shared library of ../shared (ARCHFLAGS empty and
#  -mcpu=apple-m4); the three libraries belong in the same directory.
#  libulmblas.dylib exports the functions of the backends (generated
#  trampolines) and ulmblas_backend() (the selected backend, "neon" or
#  "sme").  Not forwarded: the two variables of the reference CBLAS
#  (RowMajorStrg, CBLAS_CallFromC), and the error handlers xerbla_ and
#  cblas_xerbla: a handler defined by the program is passed to the
#  backend (ulmblas_set_f77_xerbla, ulmblas_set_cblas_xerbla), otherwise
#  the backend uses its own.
#
TOPDIR = ../../..

include $(TOPDIR)/make.inc

OUTDIR     ?= $(TOPDIR)
BUILD_DIR  ?= build.dispatch

SME_ARCHFLAGS = -mcpu=apple-m4

NEON_LIB     = $(OUTDIR)/libulmblas_neon.dylib
SME_LIB      = $(OUTDIR)/libulmblas_sme.dylib
DISPATCH_LIB = $(OUTDIR)/libulmblas.dylib

.DELETE_ON_ERROR :
.PHONY : all clean FORCE

all : $(DISPATCH_LIB) $(NEON_LIB) $(SME_LIB)

$(NEON_LIB) : FORCE
	$(MAKE) -C ../shared SHARED_NAME=ulmblas_neon ARCHFLAGS= \
	        OUTDIR=$(abspath $(OUTDIR)) BUILD_DIR=$(abspath $(BUILD_DIR))/neon

$(SME_LIB) : FORCE
	$(MAKE) -C ../shared SHARED_NAME=ulmblas_sme \
	        ARCHFLAGS="$(SME_ARCHFLAGS)" \
	        OUTDIR=$(abspath $(OUTDIR)) BUILD_DIR=$(abspath $(BUILD_DIR))/sme

#  The functions of a backend (text symbols), without the error handlers
#  and the functions that set them;
#  both backends must export the same functions
$(BUILD_DIR)/functions.txt : $(NEON_LIB) $(SME_LIB)
	@mkdir -p $(BUILD_DIR)
	nm -gU $(NEON_LIB) | awk '$$2 == "T" {print $$3}' \
	    | grep -v -E '^_(xerbla_|cblas_xerbla|ulmblas_set_f77_xerbla|ulmblas_set_cblas_xerbla)$$' \
	    | sort > $@
	nm -gU $(SME_LIB) | awk '$$2 == "T" {print $$3}' \
	    | grep -v -E '^_(xerbla_|cblas_xerbla|ulmblas_set_f77_xerbla|ulmblas_set_cblas_xerbla)$$' \
	    | sort | cmp -s - $@ \
	    || { echo "error: the backends export different functions"; \
	         rm -f $@; exit 1; }

$(BUILD_DIR)/names.h $(BUILD_DIR)/trampolines.S : $(BUILD_DIR)/functions.txt \
                                                   gen.sh
	$(SHELL) gen.sh $< $(BUILD_DIR)

$(BUILD_DIR)/exports.txt : $(BUILD_DIR)/functions.txt
	{ cat $<; echo _ulmblas_backend; } > $@

$(BUILD_DIR)/dispatch.o : dispatch.c $(BUILD_DIR)/names.h
	$(CC) $(CFLAGS) -O2 -fPIC -I$(BUILD_DIR) -c -o $@ $<

$(BUILD_DIR)/trampolines.o : $(BUILD_DIR)/trampolines.S
	$(CC) -c -o $@ $<

$(DISPATCH_LIB) : $(BUILD_DIR)/dispatch.o $(BUILD_DIR)/trampolines.o \
                  $(BUILD_DIR)/exports.txt
	$(CC) $(LDFLAGS) -dynamiclib -install_name @rpath/libulmblas.dylib \
	      -Wl,-exported_symbols_list,$(BUILD_DIR)/exports.txt \
	      -o $@ $(BUILD_DIR)/dispatch.o $(BUILD_DIR)/trampolines.o

clean :
	$(RM) -r $(BUILD_DIR) $(DISPATCH_LIB) $(NEON_LIB) $(SME_LIB)
