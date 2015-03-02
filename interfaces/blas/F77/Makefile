TOPDIR = ../../..

include $(TOPDIR)/make.inc

VARIANTS = $(patsubst config/make.%.inc, config_%, $(wildcard config/make.*.inc))

CONFIGGOAL = $(if $(findstring clean,$(MAKECMDGOALS)), clean)

all : $(VARIANTS)

config_% :
	make -C . -f Makefile.config $(CONFIGGOAL) CONFIG=$(patsubst config_%,%, $@)

.PHONY : clean
clean : $(VARIANTS)
