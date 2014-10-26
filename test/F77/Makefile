FC = gfortran

SOURCE_FILES = $(wildcard *.f)
OBJECT_FILES = $(patsubst %.f, %.o, $(SOURCE_FILES))
TARGET_FILES = $(patsubst %.f, %, $(SOURCE_FILES))

LIBBLAS = ../../libblas.a

LDFLAGS = $(LIBBLAS) -lstdc++

all :  $(TARGET_FILES)

$(OBJECT_FILES): $(LIBBLAS)

check: $(TARGET_FILES)
	@set -e; $(RM) DBLAT1.SUMM; ./dblat1 > DBLAT1.SUMM; \
	echo "Results for level 1 BLAS double precision: DBLAT1.SUMM"
	@set -e; $(RM) DBLAT2.SUMM; ./dblat2 < dblat2.dat; \
	echo "Results for level 2 BLAS double precision: DBLAT2.SUMM"
	@set -e; $(RM) DBLAT3.SUMM; ./dblat3 < dblat3.dat; \
	echo "Results for level 3 BLAS double precision: DBLAT3.SUMM"

% : %.o
	$(FC) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	$(RM) $(TARGET_FILES) *.SUMM
