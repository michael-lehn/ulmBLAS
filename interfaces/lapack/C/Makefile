TOPDIR             = ../../..
CBLASMAKEFILE_PATH = ../../blas/C

include $(TOPDIR)/make.inc

VARIANTS = $(patsubst $(CBLASMAKEFILE_PATH)/config/make.%.inc, %,\
                      $(wildcard $(CBLASMAKEFILE_PATH)/config/make.*.inc))

all : $(VARIANTS)

$(VARIANTS) :
	make -C . -f Makefile.config $(MAKECMDGOALS) CONFIG=$@

.PHONY : clean
clean : $(VARIANTS)
