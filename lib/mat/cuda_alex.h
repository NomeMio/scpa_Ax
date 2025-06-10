#ifdef __cplusplus
extern "C" {
#endif
void copyVectorBackToHost(Vector *cpu, Vector *gpu);
void allocateAndCopyVector(Vector *cpu, Vector **gpu);
void freeVectorGpu(Vector **vec_gpu);
void testVectors(int rows);
int multCudaCSRKernelWarpCoal(MatriceCsr *mat,Vector *vector,Vector *result,double *time,unsigned int threadsPerBlock);
int multCudaCSRKernelWarp(MatriceCsr *mat,Vector *vector,Vector *result,double *time,unsigned int threadsPerBlock);
int multCudaCSRKernelLinear(MatriceCsr *mat,Vector *vector,Vector *result,double *time,unsigned int threadsPerBlock);
int multCudaCSRKernelMiniWarp(MatriceCsr *mat,Vector *vector,Vector *result,double *time,int miniWarpSize,int righePerBlocco);
int multCudaCSRKernelMiniWarpShuffle(MatriceCsr *mat,Vector *vector,Vector *result,double *time,int  miniWarpSize,int righePerBlocco);
#ifdef __cplusplus
}
#endif