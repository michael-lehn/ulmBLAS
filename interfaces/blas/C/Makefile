include ../../../make.inc

SOURCE_FILES     = $(wildcard *.cc)
DEP_FILES        = $(patsubst %.cc, %.d, $(SOURCE_FILES))

ATLAS_PREFIX     = atl_
OBJECT_FILES     = $(patsubst %.cc, %.o,                $(SOURCE_FILES))
ATL_OBJECT_FILES = $(patsubst %.cc, $(ATLAS_PREFIX)%.o, $(SOURCE_FILES))

CXXFLAGS += -I ../../..

CBLAS   = ../../../libcblas.a
ATLBLAS = ../../../libatlblas.a

all : $(CBLAS) $(ATLBLAS)

$(CBLAS) : $(OBJECT_FILES)
	ar cru $(CBLAS) $(OBJECT_FILES)
	ranlib $(CBLAS)

$(ATLBLAS) : $(ATL_OBJECT_FILES)
	ar cru $(ATLBLAS) $(ATL_OBJECT_FILES)
	ranlib $(ATLBLAS)

atl_%.o : %.cc
	$(CXX) $(CXXFLAGS) -DFAKE_ATLAS -c -o $@ $<

%.d : %.cc
	@set -e; $(CXX) $(CXXFLAGS) -MM $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $(ATLAS_PREFIX)\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$

-include $(DEP_FILES)

clean :
	$(RM) $(DEP_FILES)
	$(RM) $(OBJECT_FILES)
	$(RM) $(ATL_OBJECT_FILES)
	$(RM) $(CBLAS)
	$(RM) $(ATLBLAS)
