#ifdef __cplusplus
extern "C" {
#endif

void freeFlatHLL(FlatELLMatrix **flatMat);
int convertHLLToFlatELL(MatriceHLL **H, FlatELLMatrix **flatMat);
void printFlatELLMatrix(FlatELLMatrix **flatMat);
int invokeKernel1( Vector *vect,
     Vector *result,
     FlatELLMatrix *cudaHllMat,  MatriceHLL *matHll,int hack, double *time,int blockS );
int invokeKernel2( Vector *vect,
     Vector *result,
     FlatELLMatrix *cudaHllMat,  MatriceHLL *matHll,int hack, double *time,int blockS );
int invokeKernel3( Vector *vect,
     Vector *result,
     FlatELLMatrix *cudaHllMat,  MatriceHLL *matHll,int hack, double *time,int blockS );
     int invokeKernelWarpColonne(struct Vector *vect,
    struct Vector *result,
    struct FlatELLMatrix *cudaHllMat, struct MatriceHLL *matHll,int hack,double* time,int blockS );
    
#ifdef __cplusplus
}
#endif