#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "matriciOpp.h"
#include "cuda_luca.h"
#include <time.h>
#include <omp.h>
#define DEBUG 0


int convertRawToHll(struct MatriceRaw *matricePointer, int hackSizeP, struct MatriceHLL **hllP)
{
    int totalRows = matricePointer->height;
    int totalCols = matricePointer->width;
    int hackSize = hackSizeP;
    int numBlocks = (totalRows + hackSize - 1) / hackSize; // arrotondamento per eccesso

#if DEBUG == 1
    printf("DEBUG: totalRows=%d, totalCols=%d, hackSize=%d, numBlocks=%d\n", totalRows, totalCols, hackSize, numBlocks);
#endif
    // Allocazione della struttura HLL
    struct MatriceHLL *hll = malloc(sizeof(struct MatriceHLL));
    if (hll == NULL)
    {
        fprintf(stderr, "Errore di allocazione per MatriceHLL.\n");
        return -1;
    }
    *hllP = hll;

    hll->totalRows = totalRows;
    hll->totalCols = totalCols;
    hll->HackSize = hackSize;
    hll->numBlocks = numBlocks;

    hll->blocks = malloc(numBlocks * sizeof(ELLPACK_Block *));
    if (hll->blocks == NULL)
    {
        fprintf(stderr, "Errore di allocazione per l'array dei blocchi.\n");
        free(hll);
        return -2;
    }
    for (int blk = 0; blk < numBlocks; blk++)
    {
        hll->blocks[blk] = malloc(sizeof(ELLPACK_Block));
        if (hll->blocks[blk] == NULL)
        {
            fprintf(stderr, "Errore di allocazione per il blocco %d.\n", blk);
            // Liberare quelli allocati finora e hll
            return -3;
        }
    }

    // Per ogni blocco...
    for (int blk = 0; blk < numBlocks; blk++)
    {
#if DEBUG == 1
        printf("DEBUG: Elaborazione blocco %d\n", blk);
#endif
        struct MatriceRaw blockRaw;
        blockRaw.width = matricePointer->width;

        int row_start = blk * hackSize;
        int row_end = (blk + 1) * hackSize;
        if (row_end > totalRows)
            row_end = totalRows;
        int blockRows = row_end - row_start;
        blockRaw.height = blockRows;
#if DEBUG == 1
        printf("DEBUG: Bloc %d, row_start=%d, row_end=%d, blockRows=%d\n", blk, row_start, row_end, blockRows);
#endif
        // Conta i valori appartenenti al blocco
        int count = 0;
        for (int k = 0; k < matricePointer->nz; k++)
        {
            int i = matricePointer->iVettore[k];
            if (i >= row_start && i < row_end)
            {
                count++;
            }
        }
        // printf("DEBUG: Bloc %d, count non-zero elementi = %d\n", blk, count);

        blockRaw.nz = count;
        blockRaw.iVettore = malloc(count * sizeof(int));
        blockRaw.jVettore = malloc(count * sizeof(int));
        blockRaw.valori = malloc(count * sizeof(double));
        if (blockRaw.iVettore == NULL || blockRaw.jVettore == NULL || blockRaw.valori == NULL)
        {
            fprintf(stderr, "Errore di allocazione per il blocco RAW del blocco %d.\n", blk);
            free(blockRaw.iVettore);
            free(blockRaw.jVettore);
            free(blockRaw.valori);
            return -3;
        }

        int idx = 0;
        for (int k = 0; k < matricePointer->nz; k++)
        {
            int i = matricePointer->iVettore[k];
            if (i >= row_start && i < row_end)
            {
                blockRaw.iVettore[idx] = i - row_start; // shift per il blocco locale
                blockRaw.jVettore[idx] = matricePointer->jVettore[k];
                blockRaw.valori[idx] = matricePointer->valori[k];
                idx++;
            }
        }
        if (idx != count)
        {
            fprintf(stderr, "Errore: idx (%d) diverso da count (%d) nel blocco %d.\n", idx, count, blk);
            return -4;
        }
#if DEBUG == 1
        printf("DEBUG: Bloc %d, blockRaw.nz = %d\n", blk, blockRaw.nz);
#endif
        // Converti il blocco RAW in ELLPACK_Block
        int err = convertRawToEllpack(&blockRaw, blockRows, &hll->blocks[blk]);
        if (err != 0)
        {
            fprintf(stderr, "Errore nella conversione del blocco %d: codice %d\n", blk, err);
            return err;
        }
#if DEBUG == 1
        printf("DEBUG: Bloc %d, conversione in ELLPACK completata.\n", blk);
#endif
        free(blockRaw.iVettore);
        free(blockRaw.jVettore);
        free(blockRaw.valori);
    }

    return 1; // Successo
}

int convertRawToHll2(struct MatriceRaw *matricePointer, int hackSizeP, struct MatriceHLL **hllP)
{
    int totalRows = matricePointer->height;
    int totalCols = matricePointer->width;
    int hackSize = hackSizeP;
    int numBlocks = (totalRows + hackSize - 1) / hackSize; // arrotondamento per eccesso
#if DEBUG == 0
    printf("DEBUG: totalRows=%d, totalCols=%d, hackSize=%d, numBlocks=%d\n", totalRows, totalCols, hackSize, numBlocks);
#endif
    // Allocazione della struttura HLL
    struct MatriceHLL *hll = malloc(sizeof(struct MatriceHLL));
    if (hll == NULL)
    {
        fprintf(stderr, "Errore di allocazione per MatriceHLL.\n");
        return -1;
    }
    *hllP = hll;

    hll->totalRows = totalRows;
    hll->totalCols = totalCols;
    hll->HackSize = hackSize;
    hll->numBlocks = numBlocks;
    hll->blocks = malloc(numBlocks * sizeof(ELLPACK_Block *));
    //unsigned int  *maxNN=calloc(numBlocks,sizeof(unsigned int));
    if (hll->blocks == NULL)
    {
        fprintf(stderr, "Errore di allocazione per l'array dei blocchi.\n");
        free(hll);
        return -2;
    }
    unsigned int blockATT=0;
    int k=0;
    unsigned int nz=matricePointer->nz;
    while(blockATT<numBlocks){
            int maxNN=0;
            int indexI=k;

            for(int riga=0;riga<hackSize;riga++){
                int maxL=0;
                while(1){
                    if( indexI>=nz || matricePointer->iVettore[indexI]!=riga+blockATT*hackSize ) break;
                    maxL+=1;
                    indexI++;
                }
                maxNN=maxL>maxNN?maxL:maxNN;
            }
            //unsigned int index=k/hackSize;
            hll->blocks[blockATT] = malloc(sizeof(ELLPACK_Block));
            if (hll->blocks[blockATT] == NULL)
            {
            fprintf(stderr, "Errore di allocazione per il blocco %d.\n", blockATT);
            // Liberare quelli allocati finora e hll
            return -3;
            }
            ELLPACK_Block *b=hll->blocks[blockATT];
            b->MAXNZ=maxNN;
            b->M=hackSize;
            b->N=matricePointer->width;
            //printf("blockATT %d maxnn %d\n",blockATT,maxNN);
            b->JA=calloc(maxNN*hackSize,sizeof(int));
            b->AS=calloc(maxNN*hackSize,sizeof(double));
            if ( b->JA == NULL || b->AS == NULL)
            {
                fprintf(stderr, "Errore di allocazione per AS JA.\n");
                return -1;
            }

            for(int riga=0;riga<hackSize;riga++){
                int tempCounter=0;
                if(k>=nz){
                   b->M=riga;
                   break; 
                }              
                while(1){
                    if(k>=nz || matricePointer->iVettore[k]!=riga+blockATT*hackSize ) break;
                    b->JA[riga*maxNN+tempCounter]=matricePointer->jVettore[k];
                    b->AS[riga*maxNN+tempCounter]=matricePointer->valori[k];
                    k++;
                    tempCounter++;
                }
            }
        blockATT++;
        }
    printf("finished parsing hll matrix\n");
    return 1;
}


int convertRawToEllpack(struct MatriceRaw *matricePointer, int acksize, ELLPACK_Block **block_)
{
    int cols = matricePointer->width;
    int nz = matricePointer->nz;
    int blockRows = acksize; // acksize rappresenta il numero di righe effettive del blocco
#if DEBUG == 1
    printf("DEBUG: cols=%d, nz=%d, blockRows=%d, \n", cols, nz, blockRows);
#endif
    // Dereferenzia il puntatore al blocco
    ELLPACK_Block *block = *block_;

    // Calcola il numero di non-zero per ogni riga del blocco
    int *row_nnz = calloc(blockRows, sizeof(int));
    if (row_nnz == NULL)
    {
        fprintf(stderr, "Errore di allocazione per row_nnz.\n");
        return -1;
    }
    for (int k = 0; k < nz; k++)
    {
        int i = matricePointer->iVettore[k]; // gli indici devono essere locali al blocco, cioè in [0, blockRows)
        if (i < 0 || i >= blockRows)
        {
#if DEBUG == 1
            fprintf(stderr, "DEBUG ERROR: indice %d fuori dal range [0, %d) in posizione k=%d\n", i, blockRows, k);
#endif
            free(row_nnz);
            return -1;
        }
        row_nnz[i]++;
    }

    // Trova il massimo numero di non-zero per riga (MAXNZ)
    int MAXNZ = 0;
    for (int i = 0; i < blockRows; i++)
    {
        if (row_nnz[i] > MAXNZ)
        {
            MAXNZ = row_nnz[i];
        }
    }
#if DEBUG == 1
    printf("DEBUG: MAXNZ=%d\n", MAXNZ);
#endif
    // Se non ci sono elementi, MAXNZ sarà 0; potresti voler gestire questo caso se necessario.

    // Allocazione della struttura ELLPACK per il blocco
    block->M = blockRows;
    block->N = cols;
    block->MAXNZ = MAXNZ;
#if DEBUG == 1
    printf("DEBUG:N blocchi block=%d\n", block->M);
#endif

    block->JA = calloc(blockRows *MAXNZ ,sizeof(int));
    block->AS = calloc(blockRows * MAXNZ,sizeof(double));
    if (block->JA == NULL || block->AS == NULL)
    {
        fprintf(stderr, "Errore di allocazione per JA o AS.\n");
        free(row_nnz);
        return -1;
    }
    /*for (int i = 0; i < blockRows; i++)
    {
        //block->JA[i] = calloc(MAXNZ, sizeof(int));    // padding con zeri (o -1 se preferisci)
        //block->AS[i] = calloc(MAXNZ, sizeof(double)); // padding con zeri
        if (block->JA[i] == NULL || block->AS[i] == NULL)
        {
            fprintf(stderr, "Errore di allocazione per riga %d.\n", i);
            free(row_nnz);
            return -1;
        }
    }*/

    // Riempie la struttura ELLPACK: per ogni riga, inserisce gli indici e i valori non-zero
    int *filled = calloc(blockRows, sizeof(int));
    if (filled == NULL)
    {
        fprintf(stderr, "Errore di allocazione per filled.\n");
        free(row_nnz);
        return -1;
    }
    for (int k = 0; k < nz; k++)
    {
        int i = matricePointer->iVettore[k]; // indice locale della riga (deve essere in [0, blockRows))
        if (i < 0 || i >= blockRows)
        {
#if DEBUG == 1
            fprintf(stderr, "DEBUG ERROR: indice %d fuori dal range [0, %d) (fase riempimento) in posizione k=%d\n", i, blockRows, k);
#endif
            free(row_nnz);
            free(filled);
            return -1;
        }
        int pos = filled[i]; // posizione successiva libera nella riga i
        if (pos < 0 || pos >= MAXNZ)
        {
#if DEBUG == 1
            fprintf(stderr, "DEBUG ERROR: pos %d fuori dal range [0, %d) per riga %d\n", pos, MAXNZ, i);
#endif
            free(row_nnz);
            free(filled);
            return -1;
        }
        int j = matricePointer->jVettore[k];
        double val = matricePointer->valori[k];

        block->JA[i * MAXNZ + pos] = j;
        block->AS[i * MAXNZ + pos] = val;
        filled[i]++;

      //  printf("[DEBUG] k=%d, i=%d, pos=%d, j=%d, val=%f,JA=%d,AS=%g\n", k, i, pos, j, val,block->JA[i * MAXNZ + pos],  block->AS[i * MAXNZ + pos] = val);
    }

    free(row_nnz);
    free(filled);
    return 0; // successo
}

int printHLL(struct MatriceHLL **hllP)
{
    if (hllP == NULL || *hllP == NULL)
    {
        fprintf(stderr, "La struttura HLL è NULL.\n");
        return -1;
    }

    MatriceHLL *hll = *hllP;
    printf("Matrice HLL:\n");
    printf("  Total Rows: %d\n", hll->totalRows);
    printf("  Total Cols: %d\n", hll->totalCols);
    printf("  HackSize: %d\n", hll->HackSize);
    printf("  Num Blocks: %d\n", hll->numBlocks);

    // Per ogni blocco della matrice HLL
    for (int blk = 0; blk < hll->numBlocks; blk++)
    {
        printf("\nBlock %d:\n", blk);
        ELLPACK_Block *block = hll->blocks[blk];
        printf("  M (righe): %d\n", block->M);
        printf("  N (colonne): %d\n", block->N);
        printf("  MAXNZ (max non-zero per riga): %d\n", block->MAXNZ);
        int maxnz=block->MAXNZ;

        // Per ogni riga del blocco, stampo gli array JA e AS
        for (int i = 0; i < block->M; i++)
        {
            printf("    Row %d:\n", i);
            printf("      JA: ");
            for (int j = 0; j < block->MAXNZ; j++)
            {
                printf("%d ", block->JA[i*maxnz+j]);
            }
            printf("\n");

            printf("      AS: ");
            for (int j = 0; j < block->MAXNZ; j++)
            {
                printf("%f ", block->AS[i*maxnz+j]);
            }
            printf("\n");
        }
    }

    return 0;
}

int __attribute__((optimize("O3"))) serialMultiplyHLL(struct MatriceHLL *mat, struct Vector *vec, struct Vector *result)// tolto if importante perche inizializando la memoria con calloc trova 0 invece di una cosa a caso quindi non ci sono problemi
{

  if (!mat  || !vec || !result)
        return -1;


    if (vec->righe != mat->totalCols || result->righe != mat->totalRows)
        return -1;

    for (int b = 0; b < mat->numBlocks; b++)
    {
        ELLPACK_Block *block = mat->blocks[b];
        int globalRowStart = b * mat->HackSize;
        int maxnz=block->MAXNZ;
        for (int i = 0; i < block->M; i++)
        {
            double t = 0.0;
            for (int j = 0; j < block->MAXNZ; j++)
            {
       
                t += block->AS[i*maxnz+j] * vec->vettore[block->JA[i*maxnz+j]];
            }
            result->vettore[globalRowStart + i] = t;
        }
    }

    return 0; // esecuzione corretta
}



int  hllMultWithTime(int (*multiplayer)(struct MatriceHLL *, struct Vector *, struct Vector *), struct MatriceHLL *hll, struct Vector *vec, struct Vector *result, double *execTime)
{
    double t;
    t = omp_get_wtime();
    int retunrE=multiplayer(hll, vec, result);
    t = omp_get_wtime() - t;
    (*execTime) = t; // in seconds
    return retunrE;
}


int openMpMultiplyHllFlat(struct FlatELLMatrix *mat,struct Vector *vec, struct Vector *result);
int flatHllMultOpenMP(struct FlatELLMatrix *hll, struct Vector *vec, struct Vector *result, double *execTime){
    double t;
    t = omp_get_wtime();
    int retunrE=openMpMultiplyHllFlat(hll, vec, result);
    t = omp_get_wtime() - t;
    (*execTime) = t; // in seconds
    return retunrE;
}

int __attribute__((optimize("O3"))) openMpMultiplyHllFlat(struct FlatELLMatrix *mat, struct Vector *vec, struct Vector *result){
    if (!mat || !vec || !result)
        return -1;
    int hackSize = mat->hack;
    #pragma omp parallel for schedule(static)
    for (int block = 0; block < mat->numBlocks; block++) {
        int block_start = mat->block_offsets[block];
        int max_nnz_per_row = mat->block_nnz[block];
        int rowNumber = mat->block_rows[block];
       for (int j = 0; j < max_nnz_per_row; j++) {
            int flat_idx = block_start + j * rowNumber;
            #pragma omp simd
            for (int i = 0; i < rowNumber; i++) {
                  int col = mat->col_indices_flat[flat_idx+i];
                  double molt = vec->vettore[col];
                  double val = mat->values_flat[flat_idx+i];
                  result->vettore[hackSize * block + i] += val * molt;
            }
        }
    }
    return 0;
}

int __attribute__((optimize("O3"))) openMpMultiplyHLL(struct MatriceHLL *mat, struct Vector *vec, struct Vector *result)
{

    if (!mat || !vec || !result)
        return -1;


    if (vec->righe != mat->totalCols || result->righe != mat->totalRows)
        return -1;


                
                for (int b = 0; b < mat->numBlocks; b++)
                {
                    
                        ELLPACK_Block *block = mat->blocks[b];
                        int globalRowStart = b * mat->HackSize;
                        int maxnz = block->MAXNZ;


                        #pragma omp parallel for schedule(static)
                        for (int i = 0; i < block->M; i++) {
                            double t = 0.0;
                            int row_start = i * maxnz;
                            #pragma omp simd
                            for (int j = 0; j < maxnz; j++) {
                                t += block->AS[row_start + j] * vec->vettore[block->JA[row_start + j]];
                            }
                            result->vettore[globalRowStart + i] = t;
                        }
                    
                }
        
           
    return 0;
}

int freeMatHll(struct MatriceHLL **matricePointer) {
    if (matricePointer == NULL || *matricePointer == NULL) {
        // Handle the case where the pointer or the pointed-to struct is NULL
        return 1; // Or you might want to log an error and return
    }

    struct MatriceHLL *matHll = *matricePointer; // Use a local variable for clarity

    if (matHll->blocks != NULL) {
        for (int i = 0; i < matHll->numBlocks; i++) {
            if (matHll->blocks[i] != NULL) {
                if (matHll->blocks[i]->JA != NULL) {
                    free(matHll->blocks[i]->JA);
                    matHll->blocks[i]->JA = NULL;
                }
                if (matHll->blocks[i]->AS != NULL) {
                    free(matHll->blocks[i]->AS);
                    matHll->blocks[i]->AS = NULL;
                }
                free(matHll->blocks[i]);
                matHll->blocks[i] = NULL;
            }
        }
        free(matHll->blocks);
        matHll->blocks = NULL;
    }

    free(matHll);
    *matricePointer = NULL; // Set the original pointer to NULL

    return 0; // Indicate success
}
