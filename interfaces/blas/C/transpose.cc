#include BLAS_HEADER

extern "C" {

enum CBLAS_TRANSPOSE
transpose(enum CBLAS_TRANSPOSE trans)
{
    if (trans==CblasNoTrans) {
        trans = CblasTrans;
    } else if (trans==CblasTrans) {
        trans = CblasNoTrans;
    } else if (trans==CblasConjTrans) {
        trans = AtlasConj;
    } else if (trans==AtlasConj) {
        trans = CblasConjTrans;
    }
    return trans;
}

} // extern "C"
