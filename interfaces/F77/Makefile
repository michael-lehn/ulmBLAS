FC = gfortran
CXX = g++-4.8
CXXFLAGS = -std=c++11 -O3 -Wall -I ../..
FFLAGS   = -Wall

SOURCE_FILES     = $(wildcard *.cc) \
                   $(wildcard asum/*.cc)  $(wildcard asum/*.f) \
                   $(wildcard dot/*.cc)   $(wildcard dot/*.f) \
                   $(wildcard iamax/*.cc) $(wildcard iamax/*.f) \
                   $(wildcard nrm2/*.cc)  $(wildcard nrm2/*.f)
OBJECT_FILES     = $(patsubst %.f, %.o, $(patsubst %.cc, %.o, $(SOURCE_FILES)))

LIBBLAS = ../../libblas.a

all: $(LIBBLAS)

clean:
	$(RM) $(OBJECT_FILES)
	$(RM) $(LIBBLAS)

$(LIBBLAS) : $(OBJECT_FILES)
	ar cru $(LIBBLAS) $(OBJECT_FILES)
	ranlib $(LIBBLAS)
