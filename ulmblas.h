#define F77BLAS(x) x##_


#ifdef FAKE_ATLAS
#   define ULMBLAS(x) ATL_##x
#else
#   define ULMBLAS(x) ULM_x
#endif
