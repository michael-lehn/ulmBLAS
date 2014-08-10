SOURCE_FILES     = $(wildcard *.cc)
OBJECT_FILES     = $(patsubst %.cc, %.o,     $(SOURCE_FILES))
ATL_OBJECT_FILES = $(patsubst %.cc, atl_%.o, $(SOURCE_FILES))


CXX = g++-4.8
CXXFLAGS = -std=c++11 -O3  -msse3 -mfpmath=sse -fomit-frame-pointer -Wall -I ../..

CBLAS   = ../../libcblas.a
ATLBLAS = ../../libatlblas.a

all : $(CBLAS) $(ATLBLAS)

$(CBLAS) : $(OBJECT_FILES)
	ar cru $(CBLAS) $(OBJECT_FILES)
	ranlib $(CBLAS)

$(ATLBLAS) : $(ATL_OBJECT_FILES)
	ar cru $(ATLBLAS) $(ATL_OBJECT_FILES)
	ranlib $(ATLBLAS)

%.o : %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

atl_%.o : %.cc
	$(CXX) $(CXXFLAGS) -DFAKE_ATLAS -c -o $@ $<

clean :
	rm -f $(OBJECT_FILES)
	rm -f $(ATL_OBJECT_FILES)
	rm -f $(CBLAS)
	rm -f $(ATLBLAS)
