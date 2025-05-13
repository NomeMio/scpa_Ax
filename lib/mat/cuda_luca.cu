#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matriciOpp.h"
#include "cuda_alex.h"
#include "cuda_luca.h"
#define WARP_SIZE 32

#define DEBUG 0
  
void freeFlatHLL(FlatELLMatrix **flatMat){
    struct FlatELLMatrix *mat=*flatMat;
    free(mat->values_flat);
    free(mat->col_indices_flat);
    free(mat->block_offsets);
    free(mat->block_nnz);
    free(mat->block_rows);

}
int convertHLLToFlatELL(MatriceHLL **H, FlatELLMatrix **flatMat)
{
    // Allocazione della struttura FlatELLMatrix
    *flatMat = (FlatELLMatrix *)malloc(sizeof(FlatELLMatrix));
    if (!(*flatMat))
    {
        perror("Errore di allocazione della struttura FlatELLMatrix");
        return -1;
    }

    int numBlocks = (*H)->numBlocks;
    (*flatMat)->numBlocks = numBlocks;

    // Calcola il numero totale di elementi da allocare
    int total = 0;
    for (int b = 0; b < numBlocks; b++)
    {
        ELLPACK_Block *block = (*H)->blocks[b];
        if (block)
        {
            total += block->M * block->MAXNZ;
        }
    }
    (*flatMat)->total_values = total;

    // Allocazione degli array per il formato Flat ELLPACK
    (*flatMat)->values_flat = (double *)malloc(total * sizeof(double));
    (*flatMat)->col_indices_flat = (int *)malloc(total * sizeof(int));
    (*flatMat)->block_offsets = (int *)malloc(numBlocks * sizeof(int));
    (*flatMat)->block_nnz = (int *)malloc(numBlocks * sizeof(int));
    (*flatMat)->block_rows = (int *)malloc(numBlocks * sizeof(int));
    (*flatMat)->hack = (*H)->HackSize;


    if (!(*flatMat)->values_flat || !(*flatMat)->col_indices_flat ||
        !(*flatMat)->block_offsets || !(*flatMat)->block_nnz || !(*flatMat)->block_rows)
    {
        perror("Errore di allocazione negli array Flat ELLPACK");
        return -1;
    }

    int offset = 0;
    for (int b = 0; b < numBlocks; b++)
    {
        ELLPACK_Block *block = (*H)->blocks[b];
        if (!block)
            continue;

        int M = block->M;
        int MAXNZ = block->MAXNZ;

        // Salva i metadati per il blocco corrente
        (*flatMat)->block_offsets[b] = offset;
        (*flatMat)->block_nnz[b] = MAXNZ;
        (*flatMat)->block_rows[b] = M;

        // Copia dei dati: si copia in ordine colonna-per-riga.
        // L'elemento nella riga i e nella "colonna slot" j del blocco
        // viene memorizzato a: offset + j * M + i.
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < MAXNZ; j++)
            {
                int dst_idx = offset + j * M + i;
                int src_idx = i * MAXNZ + j; // Gli array JA e AS sono in ordine riga-per-riga
                (*flatMat)->values_flat[dst_idx] = block->AS[src_idx];
                (*flatMat)->col_indices_flat[dst_idx] = block->JA[src_idx];
            }
        }
        offset += M * MAXNZ;
    }
    printf("finished parsing hll flat matrix\n");

    return 0;
}




void printFlatELLMatrix(FlatELLMatrix **flatMat)
{
    if (flatMat == NULL || *flatMat == NULL)
    {
        printf("La struttura FlatELLMatrix è NULL.\n");
        return;
    }

    FlatELLMatrix *F = *flatMat;
    printf("Flat ELLPACK Matrix:\n");
    printf("Total values: %d, numBlocks: %d\n", F->total_values, F->numBlocks);

    // Scorre ciascun blocco
    for (int b = 0; b < F->numBlocks; b++)
    {
        int offset = F->block_offsets[b];
        int rows = F->block_rows[b];
        int maxnz = F->block_nnz[b];

        printf("Block %d: offset = %d, rows = %d, MAXNZ = %d\n", b, offset, rows, maxnz);

        // Per ogni riga del blocco
        for (int i = 0; i < rows; i++)
        {
            // Per ogni "slot" nella riga (fino a MAXNZ)
            for (int j = 0; j < maxnz; j++)
            {

                int idx = offset + j * rows + i;
                printf("[col=%d, val=%f] ", F->col_indices_flat[idx], F->values_flat[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
__global__ void matvec_flatell_kernel(struct FlatELLMatrix *dMat, double *x, double *y, int hack_size) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x; 

    
    if (global_row >= dMat->numBlocks * hack_size) return;

    // Trova a quale blocco appartiene questa riga
    int block_id = global_row / hack_size;
    if (block_id >= dMat->numBlocks) return;

    int block_start = dMat->block_offsets[block_id];   // Offset del blocco
    int rows_in_block = dMat->block_rows[block_id];    // Righe nel blocco

    // Riga locale nel blocco
    int local_row = global_row % hack_size;
    if (local_row >= rows_in_block) return;

    double sum = 0.0;
    int max_nnz = dMat->block_nnz[block_id];  // NNZ massimo per riga nel blocco

    // Moltiplicazione matrice-vettore per la riga corrente
    for (int j = 0; j < max_nnz; j++) {
        int col = dMat->col_indices_flat[block_start + j * rows_in_block + local_row];
        if (col >= 0) {
            sum += dMat->values_flat[block_start + j * rows_in_block + local_row] * x[col];
        }
    }

    y[global_row] = sum;
}


__global__ void matvec_flatell_kernel_2(FlatELLMatrix *dMat, double *x, double *y, int hack_size, int N) {
    extern __shared__ double shared_x[];
    int tid = threadIdx.x;
    int global_row = blockIdx.x * blockDim.x + tid;
    int block_size = blockDim.x;

    if (global_row >= dMat->numBlocks * hack_size) return;

    // Trova a quale blocco appartiene questa riga
    int block_id = global_row / hack_size;
    if (block_id >= dMat->numBlocks) return;

    int block_start = dMat->block_offsets[block_id];   // Offset del blocco
    int rows_in_block = dMat->block_rows[block_id];    // Righe nel blocco

    // Riga locale nel blocco
    int local_row = global_row % hack_size;
    if (local_row >= rows_in_block) return;

    // Caricamento di una porzione di x in memoria condivisa
    if (tid < block_size && tid < N) {
        shared_x[tid] = x[tid];
    }
    __syncthreads();

    double sum = 0.0;
    int max_nnz = dMat->block_nnz[block_id];  // NNZ massimo per riga nel blocco

   
    for (int j = 0; j < max_nnz; j++) {
        int col = dMat->col_indices_flat[block_start + j * rows_in_block + local_row];
        if (col >= 0) {
            // Accesso a x dalla shared memory se l'indice è nel range caricato
            double x_val = (col < block_size && col < N) ? shared_x[col] : x[col];
            sum += dMat->values_flat[block_start + j * rows_in_block + local_row] * x_val;
        }
    }

    y[global_row] = sum;
}




__global__ void matvec_flatell_kernel_v3(FlatELLMatrix *dMat, double *x, double *y, int hack_size,int total_row) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // ID del thread
    int warp_id = thread_id >> 5;  // Ogni warp lavora su una riga (thread_id / 32)
    int lane = thread_id & 31;     // ID del thread dentro la warp (0-31)



    if (warp_id >= total_row) return;

    // Calcolare a quale hack appartiene questa riga (ogni hack corrisponde a un blocco)
    int block_id = warp_id / hack_size;
    int local_row = warp_id % hack_size;
    int rows_in_block = dMat->block_rows[block_id];

    if (local_row >= rows_in_block) return;  // Assicurarsi che non si esca dai limiti della riga

    int block_start = dMat->block_offsets[block_id];  // Offset del blocco
    int max_nnz_per_row = dMat->block_nnz[block_id]; // Max NNZ per riga nel blocco
    double sum = 0.0;

    for (int j = lane; j < max_nnz_per_row; j += 32) {
        
        int flat_idx = block_start + j * rows_in_block + local_row;

        int col = dMat->col_indices_flat[flat_idx];

        // Controlla se è un padding (spesso indicato con col < 0)
        if (col >= 0) {
            double val = dMat->values_flat[flat_idx];
            sum += val * x[col]; // Accumula il prodotto
        }
    }

    int width=32;
    // Riduzione a livello di warp per sommare i risultati parziali
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset,width);
    }

    // Il primo thread della warp scrive il risultato finale
    if (lane == 0) {
        y[warp_id] = sum;
    }
}
__global__ void matvec_flatell_kernel_warpCOlonne(FlatELLMatrix *dMat, double *x, double *y, int hack_size,int total_blocks) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // ID del thread
    int warp_id = thread_id >> 5; 
    int lane = thread_id & 31;    
    if (warp_id >= total_blocks ) return; 
    int row=dMat->block_rows[warp_id];
    if(row<=lane)return;
    int block_start = dMat->block_offsets[warp_id];  // Offset del blocco
    int max_nnz_per_row = dMat->block_nnz[warp_id]; // Max NNZ per riga nel blocco
    double sum = 0.0;
    for (int j = 0; j < max_nnz_per_row-1  ; j += 1) {
        int flat_idx = block_start + j * row + lane;

        int col = dMat->col_indices_flat[flat_idx]; //evitabile?
        double molt=x[col];
        double val = dMat->values_flat[flat_idx];
        sum += val *  molt;// Accumula il prodotto
    }
    int flat_idx = block_start + (max_nnz_per_row-1) * row + lane;
    
    int col = dMat->col_indices_flat[flat_idx]; //evitabile?
    double molt=x[col];
    double val = dMat->values_flat[flat_idx];
    sum += val *  molt;// Accumula il prodotto
    y[hack_size*warp_id+lane]=sum;
}
int * allocVectorGpuInt(Vector *vect){
    int *d_vettore;
    int righex=vect->righe;
    cudaMalloc((void**)&d_vettore, sizeof(int) * vect->righe);
    cudaMemcpy(d_vettore, vect->vettore, sizeof(int) * vect->righe, cudaMemcpyHostToDevice);
    return d_vettore;    
}

typedef struct FlatEllAllocated{
    struct FlatELLMatrix * gpu;
    struct FlatELLMatrix * pointersToGpu;
}FlatEllAllocated;

// Funzione per controllare gli errori CUDA e terminare in caso di fallimento
static void HandleCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_CUDA_ERROR(err) (HandleCudaError(err, __FILE__, __LINE__))

double * allocVectorGpuDouble(Vector *vect){
    double *d_vettore;
    cudaError_t err;

    err = cudaMalloc((void**)&d_vettore, sizeof(double) * vect->righe);
    HANDLE_CUDA_ERROR(err);

    err = cudaMemcpy(d_vettore, vect->vettore, sizeof(double) * vect->righe, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    return d_vettore;
}

struct FlatEllAllocated allocateFlatHll(struct FlatELLMatrix *cudaHllMat){
    cudaError_t err;
    double *d_values_flat;
    err = cudaMalloc((void**)&d_values_flat, sizeof(double) * cudaHllMat->total_values);
    HANDLE_CUDA_ERROR(err);
    err = cudaMemcpy(d_values_flat, cudaHllMat->values_flat, sizeof(double) * cudaHllMat->total_values, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    int *d_col_indices_flat;
    err = cudaMalloc((void**)&d_col_indices_flat, sizeof(int) * cudaHllMat->total_values);
    HANDLE_CUDA_ERROR(err);
    err = cudaMemcpy(d_col_indices_flat, cudaHllMat->col_indices_flat, sizeof(int) * cudaHllMat->total_values, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    int *d_block_offsets;
    err = cudaMalloc((void**)&d_block_offsets, sizeof(int) * cudaHllMat->numBlocks);
    HANDLE_CUDA_ERROR(err);
    err = cudaMemcpy(d_block_offsets, cudaHllMat->block_offsets, sizeof(int) * cudaHllMat->numBlocks, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    int *d_block_nnz;
    err = cudaMalloc((void**)&d_block_nnz, sizeof(int) * cudaHllMat->numBlocks);
    HANDLE_CUDA_ERROR(err);
    err = cudaMemcpy(d_block_nnz, cudaHllMat->block_nnz, sizeof(int) * cudaHllMat->numBlocks, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    int *d_block_rows;
    err = cudaMalloc((void**)&d_block_rows, sizeof(int) * cudaHllMat->numBlocks);
    HANDLE_CUDA_ERROR(err);
    err = cudaMemcpy(d_block_rows, cudaHllMat->block_rows, sizeof(int) * cudaHllMat->numBlocks, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    struct FlatELLMatrix *cudaHllMatG;
    cudaHllMatG = (struct FlatELLMatrix *)malloc(sizeof(struct FlatELLMatrix));
    if (cudaHllMatG == NULL) {
        fprintf(stderr, "Errore di allocazione memoria host per cudaHllMatG in %s at line %d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaHllMatG->values_flat    = d_values_flat;
    cudaHllMatG->col_indices_flat = d_col_indices_flat;
    cudaHllMatG->block_offsets  = d_block_offsets;
    cudaHllMatG->block_nnz      = d_block_nnz;
    cudaHllMatG->block_rows     = d_block_rows;
    cudaHllMatG->hack           = cudaHllMat->hack;
    cudaHllMatG->total_values   = cudaHllMat->total_values;
    cudaHllMatG->numBlocks      = cudaHllMat->numBlocks;

    struct FlatELLMatrix *d_mat;
    err = cudaMalloc((void**)&d_mat, sizeof(struct FlatELLMatrix));
    HANDLE_CUDA_ERROR(err);
    err = cudaMemcpy(d_mat, cudaHllMatG, sizeof(struct FlatELLMatrix), cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    return (struct FlatEllAllocated) {.gpu=d_mat,.pointersToGpu=cudaHllMatG};
}


int invokeKernel1(struct Vector *vect,
        struct Vector *result,
        struct FlatELLMatrix *cudaHllMat, struct MatriceHLL *matHll,int hack, double *time ,int blockS){ //TODO:da togliere matrice hll

        cudaEvent_t start,stop;
        
    // alloco le strutture cuda
        struct FlatEllAllocated d_mat=allocateFlatHll(cudaHllMat);
        double *d_result_vettore=allocVectorGpuDouble(result);
        double *d_vettore=allocVectorGpuDouble(vect);
                
    // calcolo i blocchi necessari per l'inferenza
        int block_size = blockS;
        int num_threads = matHll->numBlocks * hack;
        int grid_size = (num_threads + block_size - 1) / block_size;
    
        // startp il timer
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        matvec_flatell_kernel<<<grid_size, block_size>>>(d_mat.gpu,d_vettore,d_result_vettore,hack);

        cudaError err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Errore nel lancio del kernel: %s\n", cudaGetErrorString(err));
            return -1;
        }
        // calcolo il tempo del esecuznione
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        double time_sec = time_ms / 1000.0;
        *time=time_sec;


        cudaError memcopy;
        memcopy=cudaMemcpy(result->vettore, d_result_vettore, result->righe * sizeof(double), cudaMemcpyDeviceToHost);
        if (memcopy!=cudaSuccess) {
            printf("errore");
        }   

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_mat.pointersToGpu->values_flat);
        cudaFree(d_mat.pointersToGpu->col_indices_flat );
        cudaFree(d_result_vettore);
        cudaFree(d_mat.pointersToGpu->block_offsets);
        cudaFree(d_mat.pointersToGpu->block_nnz);
        cudaFree(d_mat.pointersToGpu->block_rows);
        free(d_mat.pointersToGpu);

        *time=time_sec;


        
        return 0;
    
    }

int invokeKernel2(struct Vector *vect,
    struct Vector *result,
    struct FlatELLMatrix *cudaHllMat, struct MatriceHLL *matHll,int hack,double* time,int blockS){

        cudaEvent_t start,stop;
        
    // alloco le strutture cuda
        struct FlatEllAllocated d_mat=allocateFlatHll(cudaHllMat);
        double *d_result_vettore=allocVectorGpuDouble(result);
        double *d_vettore=allocVectorGpuDouble(vect);
                
   
    int block_size = blockS;
    int num_threads = matHll->numBlocks * hack;
    int grid_size = (num_threads + block_size - 1) / block_size;
    size_t shared_mem_size = num_threads * sizeof(double);

   
   
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    matvec_flatell_kernel_2<<<grid_size, block_size,1024>>>(d_mat.gpu,d_vettore,d_result_vettore,hack,vect->righe);


   cudaError err = cudaGetLastError();
        HANDLE_CUDA_ERROR(err);
        // calcolo il tempo del esecuznione
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        double time_sec = time_ms / 1000.0;
        *time=time_sec;


        cudaError memcopy;
        memcopy=cudaMemcpy(result->vettore, d_result_vettore, result->righe * sizeof(double), cudaMemcpyDeviceToHost);
        HANDLE_CUDA_ERROR(memcopy);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_mat.pointersToGpu->values_flat);
        cudaFree(d_mat.pointersToGpu->col_indices_flat );
        cudaFree(d_result_vettore);
        cudaFree(d_mat.pointersToGpu->block_offsets);
        cudaFree(d_mat.pointersToGpu->block_nnz);
        cudaFree(d_mat.pointersToGpu->block_rows);
        free(d_mat.pointersToGpu);

        *time=time_sec;


        
        return 0;
  
}


int invokeKernel3(struct Vector *vect,
    struct Vector *result,
    struct FlatELLMatrix *cudaHllMat, struct MatriceHLL *matHll,int hack,double* time,int blockS ){

        cudaEvent_t start,stop;
        
    // alloco le strutture cuda
        struct FlatEllAllocated d_mat=allocateFlatHll(cudaHllMat);
        double *d_result_vettore=allocVectorGpuDouble(result);
        double *d_vettore=allocVectorGpuDouble(vect);
                
   
    int block_size = blockS;
    int num_threads = matHll->numBlocks * hack;
    int grid_size = (num_threads + block_size - 1) / block_size;
    size_t shared_mem_size = num_threads * sizeof(double);

   
   
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int threadsPerBlock = blockS;
    int numBlocks = matHll->totalRows;
   

    matvec_flatell_kernel_v3<<<numBlocks, threadsPerBlock>>>(d_mat.gpu,d_vettore,d_result_vettore,hack,matHll->totalRows);


     cudaError err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Errore nel lancio del kernel: %s\n", cudaGetErrorString(err));
            return -1;
        }
        // calcolo il tempo del esecuznione
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        double time_sec = time_ms / 1000.0;
        *time=time_sec;


        cudaError memcopy;
        memcopy=cudaMemcpy(result->vettore, d_result_vettore, result->righe * sizeof(double), cudaMemcpyDeviceToHost);
        if (memcopy!=cudaSuccess) {
            printf("errore");
        }   

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_mat.pointersToGpu->values_flat);
        cudaFree(d_mat.pointersToGpu->col_indices_flat );
        cudaFree(d_result_vettore);
        cudaFree(d_mat.pointersToGpu->block_offsets);
        cudaFree(d_mat.pointersToGpu->block_nnz);
        cudaFree(d_mat.pointersToGpu->block_rows);
        free(d_mat.pointersToGpu);

        *time=time_sec;


        
        return 0;

  
}

int invokeKernelWarpColonne(struct Vector *vect,
    struct Vector *result,
    struct FlatELLMatrix *cudaHllMat, struct MatriceHLL *matHll,int hack,double* time,int blockS ){
        cudaEvent_t start,stop;
        
        struct FlatEllAllocated d_mat=allocateFlatHll(cudaHllMat);
        double *d_result_vettore=allocVectorGpuDouble(result);
        double *d_vettore=allocVectorGpuDouble(vect);
                
   
    int block_size = blockS;
    int num_threads = matHll->numBlocks*32;
    int grid_size = (num_threads + block_size - 1) / block_size;

   
   
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int threadsPerBlock = blockS;
    //int numBlocks = matHll->totalRows;
    matvec_flatell_kernel_warpCOlonne<<<grid_size, threadsPerBlock>>>(d_mat.gpu,d_vettore,d_result_vettore,hack,matHll->numBlocks);


     cudaError err = cudaGetLastError();
        HANDLE_CUDA_ERROR(err);
        // calcolo il tempo del esecuznione
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        double time_sec = time_ms / 1000.0;
        *time=time_sec;


        cudaError memcopy;
        memcopy=cudaMemcpy(result->vettore, d_result_vettore, result->righe * sizeof(double), cudaMemcpyDeviceToHost);
        HANDLE_CUDA_ERROR(memcopy);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_mat.pointersToGpu->values_flat);
        cudaFree(d_mat.pointersToGpu->col_indices_flat );
        cudaFree(d_result_vettore);
        cudaFree(d_mat.pointersToGpu->block_offsets);
        cudaFree(d_mat.pointersToGpu->block_nnz);
        cudaFree(d_mat.pointersToGpu->block_rows);
        free(d_mat.pointersToGpu);

        *time=time_sec;


        
        return 0;

  
}