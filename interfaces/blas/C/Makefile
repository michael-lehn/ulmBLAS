TOPDIR = ../../..

include $(TOPDIR)/make.inc

VARIANTS = $(patsubst config/make.%.inc, %, $(wildcard config/make.*.inc))

CONFIGGOAL = $(if $(findstring clean,$(MAKECMDGOALS)), clean)

all : $(VARIANTS)

% :
	@echo VARIANTS=$(VARIANTS)
	make -C . -f Makefile.config $(CONFIGGOAL) CONFIG=$@

.PHONY : clean
clean : $(VARIANTS)
