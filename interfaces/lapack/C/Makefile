SOURCE_FILES     = $(wildcard *.cc)
DEP_FILES        = $(patsubst %.cc, %.d, $(SOURCE_FILES))

ATLAS_PREFIX     = atl_
OBJECT_FILES     = $(patsubst %.cc, %.o,                $(SOURCE_FILES))
ATL_OBJECT_FILES = $(patsubst %.cc, $(ATLAS_PREFIX)%.o, $(SOURCE_FILES))

CXX      = g++
CXXFLAGS = -std=c++11 -O3  -fomit-frame-pointer -Wall -I ../../..
CXXFLAGS += -msse3 -mfpmath=sse

CLAPACK   = ../../../libclapack.a
ATLLAPACK = ../../../libatllapack.a

all : $(CLAPACK) $(ATLLAPACK)

$(CLAPACK) : $(OBJECT_FILES)
	ar cru $(CLAPACK) $(OBJECT_FILES)
	ranlib $(CLAPACK)

$(ATLLAPACK) : $(ATL_OBJECT_FILES)
	ar cru $(ATLLAPACK) $(ATL_OBJECT_FILES)
	ranlib $(ATLLAPACK)

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
	$(RM) $(CLAPACK)
	$(RM) $(ATLLAPACK)
