TOPDIR = ..

include $(TOPDIR)/make.inc

SOURCE_FILES = $(wildcard *.f)
OBJECT_FILES = $(patsubst %.f,%.o,$(SOURCE_FILES))
REFBLAS_LIB  = $(TOPDIR)/librefblas.a

all : $(REFBLAS_LIB)

$(REFBLAS_LIB) : $(OBJECT_FILES)
	ar cru $(REFBLAS_LIB) $(OBJECT_FILES)
	ranlib $(REFBLAS_LIB)

%.o : %.f
	$(FC) $(FFLAGS) -c -o $@ $<

clean :
	rm -f $(OBJECT_FILES)
	rm -f $(REFBLAS_LIB)

