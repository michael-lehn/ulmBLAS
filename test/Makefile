FC = gfortran

SOURCE_FILES = $(wildcard *.f)
TEST_REFBLAS = $(patsubst %.f, %_ref, $(SOURCE_FILES))
TEST_ULMBLAS = $(patsubst %.f, %_ulm, $(SOURCE_FILES))

LIBREFBLAS = ../librefblas.a
LIBULMBLAS = ../libulmblas.a

REF_LDFLAGS  += -L.. -lrefblas
ULM_LDFLAGS  += -L.. -lulmblas

all :  $(LIBREFBLAS) $(LIBULMBLAS) $(TEST_REFBLAS) $(TEST_ULMBLAS)

%_ref: %.f $(LIBREFBLAS)
	$(FC) $< $(REF_LDFLAGS) -o $@

%_ulm: %.f $(LIBULMBLAS)
	$(FC) $< $(ULM_LDFLAGS) -o $@

check_ulm:
	$(RM) *.SUMM
	./dblat1_ulm
	./dblat2_ulm < dblat2.dat
	./dblat3_ulm < dblat3.dat

check_ref:
	$(RM) *.SUMM
	./dblat1_ref
	./dblat2_ref < dblat2.dat
	./dblat3_ref < dblat3.dat

clean:
	$(RM) $(TEST_REFBLAS) $(TEST_ULMBLAS) *.SUMM
