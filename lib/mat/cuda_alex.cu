#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"
#include "cuda_alex.h"
#include <cstdint>
#define DEBUG 0

int debug=0;

#if DEBUG == 1
    debug=1;
#endif

#include <cuda_runtime.h>
#include <stdio.h>
// Macro to check CUDA calls
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA ERROR: %s (Error Code: %d) at %s:%d\n",  \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Macro for safe cudaMalloc
#define CUDA_MALLOC(ptr, size)                                  \
    do {                                                         \
        CUDA_CHECK(cudaMalloc((void**)&ptr, (size)));            \
            if(debug==1)fprintf(stderr,"Allocated %lu bytes at %p [%s:%d]\n",        \
                   (size_t)(size), (void*)(ptr), __FILE__, __LINE__); \
    } while (0)


// Macro for safe cudaFree
#define CUDA_FREE(ptr)                                        \
    do {                                                     \
        if ((ptr) != NULL) {                                 \
            CUDA_CHECK(cudaFree(ptr));                       \
            if(debug==1)fprintf(stderr,"Freed memory at %p [%s:%d]\n",           \
                   (void*)(ptr), __FILE__, __LINE__);        \
            (ptr) = NULL;  /* Avoid dangling pointer */      \
        }                                                   \
    } while (0)

// Macro for safe cudaMemcpy
#define CUDA_MEMCPY(dst, src, size, direction)                           \
    do {                                                                 \
        CUDA_CHECK(cudaMemcpy((dst), (src), (size), (direction)));       \
        if(debug==1)fprintf(stderr,"[CUDA MEMCPY] %lu bytes from %p to %p (Dir: %d) [%s:%d]\n", \
               (size_t)(size), (void*)(src), (void*)(dst), (direction), __FILE__, __LINE__); \
    } while (0)


void checkerror(){
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
    }
}
// Macro for safe cudaMemcpy
#define CUDA_TIME(operation,time)                           \
    { \
    cudaEvent_t start, stop;                               \
    CUDA_CHECK(cudaEventCreate(&start));                \
    CUDA_CHECK(cudaEventCreate(&stop));                 \
    CUDA_CHECK(cudaEventRecord(start, 0));              \
    operation; \
    CUDA_CHECK(cudaEventRecord(stop, 0)); \
    CUDA_CHECK(cudaDeviceSynchronize());\
    float seconds = 0;  \
    CUDA_CHECK(cudaEventElapsedTime(&seconds, start, stop)); \
    seconds=seconds/1000.0;\
    CUDA_CHECK(cudaEventDestroy(start));\
    CUDA_CHECK(cudaEventDestroy(stop));\
    (*time)=seconds;\
    } 



void allocateAndCopyMatriceCsrGpu( MatriceCsr *orgi,  MatriceCsr **mat, int coal) {
        cudaMalloc((void**)mat, sizeof(MatriceCsr));
    
        unsigned int *d_iRP = NULL;
        unsigned int *d_jValori = NULL;
        double *d_valori = NULL;
    
        size_t size_iRP = (orgi->height + 1) * sizeof(unsigned int);
        if(coal==1){
            size_iRP=(orgi->height*2 ) * sizeof(unsigned int);
        }
        size_t size_jValori = orgi->nz * sizeof(unsigned int);
        size_t size_valori = orgi->nz * sizeof(double);
    
        CUDA_MALLOC(d_iRP, size_iRP);
        CUDA_MALLOC(d_jValori, size_jValori);
        CUDA_MALLOC(d_valori, size_valori);
    
        CUDA_MEMCPY(d_iRP, orgi->iRP, size_iRP, cudaMemcpyHostToDevice);
        CUDA_MEMCPY(d_jValori, orgi->jValori, size_jValori, cudaMemcpyHostToDevice);
        CUDA_MEMCPY(d_valori, orgi->valori, size_valori, cudaMemcpyHostToDevice);
    
        MatriceCsr temp = *orgi;
        temp.iRP = d_iRP;
        temp.jValori = d_jValori;
        temp.valori = d_valori;
    
        CUDA_MEMCPY(*mat, &temp, sizeof(MatriceCsr), cudaMemcpyHostToDevice);
}


void freeMatriceCsrGpu(MatriceCsr **mat_gpu) {
    if (mat_gpu == NULL || *mat_gpu == NULL) {
        return;
    }

    MatriceCsr temp_host;

    CUDA_CHECK(cudaMemcpy(&temp_host, *mat_gpu, sizeof(MatriceCsr), cudaMemcpyDeviceToHost));

    CUDA_FREE(temp_host.iRP);
    CUDA_FREE(temp_host.jValori);
    CUDA_FREE(temp_host.valori);

    MatriceCsr *ptr_to_free = *mat_gpu;
    CUDA_FREE(ptr_to_free);

    *mat_gpu = NULL;
}


void copyVectorBackToHost(Vector *cpu, Vector *gpu) {
    Vector temp;
    CUDA_MEMCPY(&temp, gpu, sizeof(Vector), cudaMemcpyDeviceToHost);

    size_t size_vettore = cpu->righe * sizeof(double);
    CUDA_MEMCPY(cpu->vettore, temp.vettore, size_vettore, cudaMemcpyDeviceToHost);
}
void allocateAndCopyVector(Vector *cpu, Vector **gpu) {
    CUDA_CHECK(cudaMalloc((void**)gpu, sizeof(Vector)));

    double *d_vettore = NULL;
    size_t size_vettore = cpu->righe * sizeof(double);
    CUDA_MALLOC(d_vettore, size_vettore);
    CUDA_MEMCPY(d_vettore, cpu->vettore, size_vettore, cudaMemcpyHostToDevice);

    Vector temp = *cpu;
    temp.vettore = d_vettore;

    CUDA_MEMCPY(*gpu, &temp, sizeof(Vector), cudaMemcpyHostToDevice);
}
void freeVectorGpu(Vector **vec_gpu) {
    if (vec_gpu == NULL || *vec_gpu == NULL) {
        return;
    }

    Vector temp_host;

    CUDA_CHECK(cudaMemcpy(&temp_host, *vec_gpu, sizeof(Vector), cudaMemcpyDeviceToHost));

    CUDA_FREE(temp_host.vettore);

    Vector *ptr_to_free = *vec_gpu;
    CUDA_FREE(ptr_to_free);

    *vec_gpu = NULL;
}


void vectorMultiplySerial(Vector *a, Vector* b, Vector * result) {
    for (int i = 0; i < a->righe; ++i) {
        result->vettore[i] = a->vettore[i] * b->vettore[i];
    }
}

__global__ void csr_matvec_mul(MatriceCsr *d_mat, Vector *d_vec, Vector *d_result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < d_mat->height) {
        double sum = 0.0;
        for (int j = d_mat->iRP[row]; j < d_mat->iRP[row + 1]; j++) {
            sum += d_mat->valori[j] * d_vec->vettore[d_mat->jValori[j]];
        }
        d_result->vettore[row] = sum;
    }
}


__inline__ __device__
double warpReduceSum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void crs_mat_32_way(MatriceCsr *d_mat, Vector *d_vec, Vector *d_result) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; // id
    int realRow = id >> 5; // Check row number dividing id % number of thread per warp 2^5
    int position = id & 31; // get position inside warp

    if (realRow >= d_mat->height) return; //exit if id is outisde of lines range
    int base = d_mat->iRP[realRow]; //start of array
    int rowDim = d_mat->iRP[realRow + 1] - base;
    double sum = 0.0;
    #pragma unroll
    for (int i = 0; (i + 1) * 32 <= rowDim; ++i) {
        int col_index = d_mat->jValori[base + i * 32 + position];
        double  matVal= d_mat->valori[base + i * 32 + position];
        double vectVal= d_vec->vettore[col_index];
        sum += matVal*vectVal;
    }

    int remaining = rowDim % 32;
    if (remaining > 0) {
        int start_of_remaining = base + rowDim - remaining;
        if (position < remaining) {
            int col_index = d_mat->jValori[start_of_remaining + position];
            double  matVal= d_mat->valori[start_of_remaining + position];
            double vectVal= d_vec->vettore[col_index];
            sum += matVal*vectVal;
        }
    }
    sum = warpReduceSum(sum);
    if (position == 0) {
        d_result->vettore[realRow] = sum; // Explicit cast if d_result is float
    }
}



// csr_mat mult with 8 threads per row
__global__ void crs_mat_miniWarp(MatriceCsr *d_mat, Vector *d_vec, Vector *d_result,int  miniWarpSize) {
    extern  __shared__ double reduceVals[]; 
    int id = blockIdx.x * blockDim.x + threadIdx.x; // id
    int realRow = id /miniWarpSize; // Check row number dividing id % number of thread per warp 2^3

    int position = id % miniWarpSize ;
    //int position = id & 7; // get position inside 8 threads block
  
    if (realRow >= d_mat->height) return; //exit if id is outisde of lines range
    int base = d_mat->iRP[realRow]; //start of array
    int rowDim = d_mat->iRP[realRow + 1] - base;
    double sum = 0.0;
    //#pragma unroll
    for (int i = 0; (i + 1) * miniWarpSize <= rowDim; ++i) {
        int col_index = d_mat->jValori[base + i * miniWarpSize + position];
        double  matVal= d_mat->valori[base + i * miniWarpSize + position];
        double vectVal= d_vec->vettore[col_index];
        sum += matVal*vectVal;
    }
    __syncthreads();
    int remaining = rowDim % miniWarpSize;
    if (remaining > 0) {
        int start_of_remaining = base + rowDim - remaining;
        if (position < remaining) {
            int col_index = d_mat->jValori[start_of_remaining + position];
            double  matVal= d_mat->valori[start_of_remaining + position];
            double vectVal= d_vec->vettore[col_index];
            sum += matVal*vectVal;
        }
    }
    reduceVals[threadIdx.x]=sum;
    __syncthreads();
    for(unsigned int s=miniWarpSize>>1; s>0; s >>=1) {
        if(position<s) {
            reduceVals[threadIdx.x]+=reduceVals[threadIdx.x+s];
        }
        __syncthreads();
    }
    
    if (position == 0) {
        d_result->vettore[realRow] = reduceVals[threadIdx.x]; // Explicit cast if d_result is float
    }
}



__global__ void crs_mat_32_way_coal(MatriceCsr * __restrict__ d_mat, Vector * __restrict__ d_vec, Vector *d_result) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; 
    int realRow = id >> 5; 
    const int2* input_vec_ptr = reinterpret_cast<const int2*>(d_mat->iRP);
    int2 loaded_ints = input_vec_ptr[realRow];
    int base = loaded_ints.x; //start of array
    int rowDim = loaded_ints.y ;
    if (realRow >= d_mat->height) return;
    double sum = 0.0;
    int position = id & 31;
    rowDim-= base;
    for (int i = 0; (i + 1) * 32 <= rowDim; ++i) {
        int col_index = d_mat->jValori[base + i * 32 + position];
        double  matVal= d_mat->valori[base + i * 32 + position];
        double vectVal= d_vec->vettore[col_index];
        sum += matVal*vectVal;
    }

    int remaining = rowDim % 32;
    if (remaining > 0) {
        int start_of_remaining = base + rowDim - remaining;
        if (position < remaining) {
            int col_index = d_mat->jValori[start_of_remaining + position];
            double  matVal= d_mat->valori[start_of_remaining + position];
            double vectVal= d_vec->vettore[col_index];
            sum += matVal*vectVal;
        }
    }
    sum = warpReduceSum(sum);
    if (position == 0) {
        d_result->vettore[realRow] = sum; // Explicit cast if d_result is float
    }
}


__global__ void vectorMultiply(Vector *a, Vector *b, Vector *result) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a->righe) {
        result->vettore[idx] = a->vettore[idx] * b->vettore[idx];
    }
}



int multCudaCSRKernelWarp(MatriceCsr *mat,Vector *vector,Vector *result,double *time,unsigned int threadsPerBlock){
    
    MatriceCsr *matG;
    allocateAndCopyMatriceCsrGpu(mat,&matG,0);
    Vector *vectorG;
    Vector *resultG;
    allocateAndCopyVector(vector,&vectorG);
    allocateAndCopyVector(result,&resultG);

    unsigned int rows=mat->height;
    int N = vector->righe*32;
    N=N+(threadsPerBlock-N%threadsPerBlock);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
   
    CUDA_TIME((crs_mat_32_way<<<blocksPerGrid,threadsPerBlock>>>(matG, vectorG, resultG)),time);
    //CUDA_CHECK(cudaDeviceSynchronize());
    
    copyVectorBackToHost(result,resultG);
    freeVectorGpu(&vectorG);
    freeVectorGpu(&resultG);
    freeMatriceCsrGpu(&matG);
}

int multCudaCSRKernelMiniWarp(MatriceCsr *mat,Vector *vector,Vector *result,double *time,int  miniWarpSize,int righePerBlocco){
    
    MatriceCsr *matG;
    allocateAndCopyMatriceCsrGpu(mat,&matG,0);
    Vector *vectorG;
    Vector *resultG;
    allocateAndCopyVector(vector,&vectorG);
    allocateAndCopyVector(result,&resultG);
    unsigned int threadsPerBlock = miniWarpSize*righePerBlocco; // 32 warps of 8 threads each
    unsigned int rows=mat->height;
    int N = vector->righe*miniWarpSize;
    unsigned int shardeMem=sizeof(double)*threadsPerBlock;
    N=N+(threadsPerBlock-N%threadsPerBlock);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
   
    CUDA_TIME((crs_mat_miniWarp<<<blocksPerGrid,threadsPerBlock,shardeMem>>>(matG, vectorG, resultG,miniWarpSize)),time);
    //CUDA_CHECK(cudaDeviceSynchronize());
    
    copyVectorBackToHost(result,resultG);
    freeVectorGpu(&vectorG);
    freeVectorGpu(&resultG);
    freeMatriceCsrGpu(&matG);
}



int multCudaCSRKernelWarpCoal(MatriceCsr *mat,Vector *vector,Vector *result,double *time,unsigned int threadsPerBlock){
    
    MatriceCsr *matG;
    allocateAndCopyMatriceCsrGpu(mat,&matG,1);
    Vector *vectorG;
    Vector *resultG;
    allocateAndCopyVector(vector,&vectorG);
    allocateAndCopyVector(result,&resultG);

    unsigned int rows=mat->height;
    int N = vector->righe*32;
    N=N+(threadsPerBlock-N%threadsPerBlock);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
   
    CUDA_TIME((crs_mat_32_way_coal<<<blocksPerGrid,threadsPerBlock>>>(matG, vectorG, resultG)),time);
    //CUDA_CHECK(cudaDeviceSynchronize());
    
    copyVectorBackToHost(result,resultG);
    freeVectorGpu(&vectorG);
    freeVectorGpu(&resultG);
    freeMatriceCsrGpu(&matG);
}


int multCudaCSRKernelLinear(MatriceCsr *mat,Vector *vector,Vector *result,double *time,unsigned int threadsPerBlock){
    
    MatriceCsr *matG;
    allocateAndCopyMatriceCsrGpu(mat,&matG,0);
    Vector *vectorG;
    Vector *resultG;
    allocateAndCopyVector(vector,&vectorG);
    allocateAndCopyVector(result,&resultG);

    unsigned int rows=mat->height;
    int N = vector->righe;
    N=N+(threadsPerBlock-N%threadsPerBlock);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
   
    CUDA_TIME((csr_matvec_mul<<<blocksPerGrid,threadsPerBlock>>>(matG, vectorG, resultG)),time);
    //CUDA_CHECK(cudaDeviceSynchronize());
    
    copyVectorBackToHost(result,resultG);
    freeVectorGpu(&vectorG);
    freeVectorGpu(&resultG);
    freeMatriceCsrGpu(&matG);
}



void testVectors(int rows){
Vector *vector1;
Vector *vector2;
Vector *result;
Vector *resultSerial;
int seed = 42;
int num=1000000;
generate_random_vector(seed, num, &vector1) ;
generate_random_vector(seed, num, &vector2) ;
generateEmpty(num,&result);
generateEmpty(num,&resultSerial);

Vector *vector1G;
Vector *vector2G;
Vector *resultG;
allocateAndCopyVector(vector1,&vector1G);
allocateAndCopyVector(vector2,&vector2G);
allocateAndCopyVector(result,&resultG);

int N = vector1->righe;
int threadsPerBlock = 32;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
//printf("%d\n",blocksPerGrid);
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));
CUDA_CHECK(cudaEventRecord(start, 0));
vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(vector1G, vector2G, resultG);
CUDA_CHECK(cudaEventRecord(stop, 0));
CUDA_CHECK(cudaDeviceSynchronize());
float seconds = 0;//in milli secondi inizialmente
CUDA_CHECK(cudaEventElapsedTime(&seconds, start, stop));
// Clean up events
seconds=seconds/1000.0;
CUDA_CHECK(cudaEventDestroy(start));
CUDA_CHECK(cudaEventDestroy(stop));
printf("giga flops %f\n",vector1->righe/(seconds*1000000000.0));
copyVectorBackToHost(result,resultG);
clock_t t;
t = clock();
vectorMultiplySerial(vector1,vector2,resultSerial);
t = clock() - t;
double execTime = ((double)t) / CLOCKS_PER_SEC; // in seconds
printf("giga flops %f\n",vector1->righe/(execTime*1000000000.0));
int areEq=areVectorsEqual(result,resultSerial);
printf("are equal(0 yes)? %d\n",areEq);
return ;

}

