TOPDIR             = ../../..
CBLASMAKEFILE_PATH = ../../blas/C

include $(TOPDIR)/make.inc

VARIANTS = $(patsubst $(CBLASMAKEFILE_PATH)/config/make.%.inc, config_%,\
                      $(wildcard $(CBLASMAKEFILE_PATH)/config/make.*.inc))

CONFIGGOAL = $(if $(findstring clean,$(MAKECMDGOALS)), clean)

all : $(VARIANTS)

config_% :
	make -C . -f Makefile.config $(CONFIGGOAL) CONFIG=$(patsubst config_%,%, $@)

.PHONY : clean
clean : $(VARIANTS)
