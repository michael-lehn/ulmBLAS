#
#  Note: bench must be the last directory.  That's because building the
#        benchmark suite requires that the BLAS libraries are already built.
#
DIRS= level1 refblas test bench

all:
	-for dir in $(DIRS); do make -C $$dir; done

clean:
	-for dir in $(DIRS); do make -C $$dir clean; done

check:
	make -C test check_ulm
