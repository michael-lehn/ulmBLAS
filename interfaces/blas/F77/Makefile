TOPDIR = ../../..

include $(TOPDIR)/make.inc

VARIANTS = $(patsubst config/make.%.inc, config_%, $(wildcard config/make.*.inc))

CONFIGGOAL = $(if $(findstring clean,$(MAKECMDGOALS)), clean)

.PHONY : all clean $(VARIANTS)

all : $(VARIANTS)

$(VARIANTS) :
	$(MAKE) -f Makefile.config $(CONFIGGOAL) CONFIG=$(patsubst config_%,%,$@)

clean : $(VARIANTS)
