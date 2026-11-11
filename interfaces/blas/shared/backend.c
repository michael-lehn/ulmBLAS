/*
 *  ulmblas_backend: the kernels of this library.  For a backend of the
 *  dispatching library (../dispatch) its name (sse3, avx2, ...; the front
 *  has its own ulmblas_backend), for a library without dispatching the
 *  architecture it was built for (aarch64, x86_64, ...).
 */
const char *
ulmblas_backend(void)
{
    return ULMBLAS_BACKEND_NAME;
}
