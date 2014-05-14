FC      = gfortran
FFLAGS  += -fimplicit-none -O3

SOURCE_FILES = $(wildcard *.f)
OBJECT_FILES = $(patsubst %.f,%.o,$(SOURCE_FILES))
REFBLAS      = ../librefblas.a

all : $(REFBLAS)

$(REFBLAS) : $(OBJECT_FILES)
	ar cru $(REFBLAS) $(OBJECT_FILES)
	ranlib $(REFBLAS)

%.o : %.f
	$(FC) $(FFLAGS) -c -o $@ $<

clean :
	rm -f $(OBJECT_FILES)
	rm -f $(REFBLAS)

