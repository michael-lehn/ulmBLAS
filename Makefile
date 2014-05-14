DIRS= level1 refblas test

all:
	-for dir in $(DIRS); do make -C $$dir; done

clean:
	-for dir in $(DIRS); do make -C $$dir clean; done

check:
	make -C test check_ulm
