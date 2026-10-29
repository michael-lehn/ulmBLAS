#
#  Netlib reference BLAS (Fortran 77).  Only used by the benchmarks.
#
TOPDIR = ..

include $(TOPDIR)/make.inc

BUILD_DIR    = build
SOURCE_FILES = $(wildcard *.f)
OBJECT_FILES = $(patsubst %.f,$(BUILD_DIR)/%.o,$(SOURCE_FILES))
REFBLAS_LIB  = $(TOPDIR)/librefblas.a

.DELETE_ON_ERROR :
.PHONY : all clean FORCE

all : $(REFBLAS_LIB)

$(REFBLAS_LIB) : $(OBJECT_FILES)
	$(call make_archive,$@,$(OBJECT_FILES))

$(BUILD_DIR) :
	mkdir -p $@

$(BUILD_DIR)/compile.cmd : FORCE | $(BUILD_DIR)
	@$(call record_command,$(FCOMPILE),$@)

$(BUILD_DIR)/%.o : %.f $(BUILD_DIR)/compile.cmd
	$(FCOMPILE) -c -o $@ $<

clean :
	rm -rf $(BUILD_DIR)
	rm -f $(REFBLAS_LIB)
