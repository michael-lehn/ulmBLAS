TOPDIR = ../../..

include $(TOPDIR)/make.inc

F77CONFIG_PATH = ../../blas/F77

VARIANTS = $(patsubst $(F77CONFIG_PATH)/config/make.%.inc, config_%, \
                      $(wildcard $(F77CONFIG_PATH)/config/make.*.inc))

CONFIGGOAL = $(if $(findstring clean,$(MAKECMDGOALS)), clean)

.PHONY : all clean $(VARIANTS)

all : $(VARIANTS)

$(VARIANTS) :
	$(MAKE) -f Makefile.config $(CONFIGGOAL) CONFIG=$(patsubst config_%,%,$@)

clean : $(VARIANTS)
