//
//  ulmBLAS: undefines the macros of compat.h at the end of a vendored BLIS
//  kernel header (kernel/*.h).
//
#undef restrict
#undef BLIS_OS_OSX
#undef ULMBLAS_BLIS_LABEL
#undef ULMBLAS_BLIS_SETUP_CT_PRE_
#undef ULMBLAS_BLIS_SETUP_CT_POST_
#undef GEMM_UKR_SETUP_CT
#undef GEMM_UKR_SETUP_CT_AMBI
#undef GEMM_UKR_SETUP_CT_ANY
#undef GEMM_UKR_SETUP_CT_ALIGNED
#undef GEMM_UKR_FLUSH_CT
#undef ULMBLAS_BLIS_CT_TYPE_s
#undef ULMBLAS_BLIS_CT_TYPE_d
#undef ULMBLAS_BLIS_CT_TYPE_c
#undef ULMBLAS_BLIS_CT_TYPE_z
#undef ULMBLAS_BLIS_CT_TYPE
#undef AOCL_DTL_TRACE_ENTRY
#undef AOCL_DTL_TRACE_EXIT
#undef AOCL_DTL_LEVEL_TRACE_7
