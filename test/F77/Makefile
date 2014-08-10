FC = gfortran

SOURCE_FILES = $(wildcard *.f)
OBJECT_FILES = $(patsubst %.f, %.o, $(SOURCE_FILES))
TARGET_FILES = $(patsubst %.f, %, $(SOURCE_FILES))

LIBBLAS = ../../libblas.a

LDFLAGS = $(LIBBLAS) -lstdc++

all :  $(TARGET_FILES)

$(OBJECT_FILES): $(LIBBLAS)

% : %.o
	$(FC) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	$(RM) $(TARGET_FILES) *.SUMM
