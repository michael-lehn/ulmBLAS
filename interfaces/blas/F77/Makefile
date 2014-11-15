include ../../../make.inc

VARIANTS = $(patsubst config/make.%.inc, config_%, $(wildcard config/make.*.inc))

all : $(VARIANTS)

config_% :
	make -C . -f Makefile.config $(MAKECMDGOALS) CONFIG=$(patsubst config_%,%, $@)

.PHONY : clean
clean : $(VARIANTS)
