
#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "matriciOpp.h"
#include <time.h>
#include <omp.h>
#include <string.h>
typedef struct TempStruct
{ // Defining a struct only for this function, for organizing the matrix in rows
    unsigned int elements;
    size_t bufferSize;
    unsigned int *indices; // preparing a list of indexes of MatriceRaw format
};

int convertRawToCsr(struct MatriceRaw *matricePointer, struct MatriceCsr **csrPointer)
{ // Devo scorrere iVec e trovare tutti gli elementi di tutte le righe

    int nRighe = matricePointer->height;
    struct TempStruct *indicesArray = malloc(sizeof(struct TempStruct) * nRighe); // Each element rappresents a line in the matrix
    if (indicesArray == NULL)
        return -1;
    for (int i = 0; i < nRighe; i++)
    {
        indicesArray[i].elements = 0;
        indicesArray[i].bufferSize = 4;
        indicesArray[i].indices = malloc(indicesArray[i].bufferSize * sizeof(unsigned int));
        if (indicesArray[i].indices == NULL)
            return -1;
    }
    for (int i = 0; i < matricePointer->nz; i++)
    { // per ogni riga aggiungo alla rispettiva struttura tutti gli indici di MatriceRaw che riguardano la riga stessa
        int indiceAttuale = matricePointer->iVettore[i];
        if (indicesArray[indiceAttuale].elements == indicesArray[indiceAttuale].bufferSize)
        {
            unsigned int *newindex = malloc(sizeof(unsigned int) * indicesArray[indiceAttuale].bufferSize * 2);
            memcpy(newindex, indicesArray[indiceAttuale].indices, sizeof(unsigned int) * indicesArray[indiceAttuale].bufferSize);
            indicesArray[indiceAttuale].bufferSize = indicesArray[indiceAttuale].bufferSize * 2;
            free(indicesArray[indiceAttuale].indices);
            indicesArray[indiceAttuale].indices = newindex;
        }
        indicesArray[indiceAttuale].indices[indicesArray[indiceAttuale].elements] = i;
        indicesArray[indiceAttuale].elements += 1;
    }
    unsigned int numeroNz = matricePointer->nz;
    *csrPointer = malloc(sizeof(struct MatriceCsr));
    if (*csrPointer == NULL)
        return -1;
    struct MatriceCsr *matrixCsr = *csrPointer;
    matrixCsr->nz = numeroNz;
    matrixCsr->height = nRighe;
    matrixCsr->width = matricePointer->width;
    matrixCsr->iRP = malloc(sizeof(unsigned int) * (nRighe + 1));
    matrixCsr->jValori = malloc(sizeof(unsigned int) * numeroNz);
    matrixCsr->valori = malloc(sizeof(double) * numeroNz);
    if (matrixCsr->iRP == NULL || matrixCsr->jValori == NULL || matrixCsr->valori == NULL)
        return -1;

    int j = 0;
    matrixCsr->iRP[nRighe] = numeroNz;
    for (int i = 0; i < nRighe; i++)
    {
        matrixCsr->iRP[i] = j;
        for (int i2 = 0; i2 < indicesArray[i].elements; i2++)
        {
            matrixCsr->jValori[j] = matricePointer->jVettore[indicesArray[i].indices[i2]];
            matrixCsr->valori[j] = matricePointer->valori[indicesArray[i].indices[i2]];
            j++;
        }
    }

    for (int i = 0; i < nRighe; i++)
    {
        free(indicesArray[i].indices);
    }
    free(indicesArray);
    return 0;
}

int __attribute__((optimize("O3"))) serialCsrMult(struct MatriceCsr *csr, struct Vector *vec, struct Vector *result)
{
    unsigned int nrows = vec->righe;
    for (int i = 0; i < nrows; i++)
    {
        double sum = 0;
        for (int j = csr->iRP[i]; j < csr->iRP[i + 1]; j++)
        {
            sum += csr->valori[j] * vec->vettore[csr->jValori[j]];
        }
        result->vettore[i] = sum;
    }
}

int __attribute__((optimize("O3"))) parallelCsrMult(struct MatriceCsr *csr, struct Vector *vec, struct Vector *result)
{
    unsigned int nrows = vec->righe;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nrows; i++)
    {
        //int thread_id = omp_get_thread_num();
        //printf("Hello from thread %d\n", thread_id);

        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = csr->iRP[i]; j < csr->iRP[i + 1]; j++)
        {
            sum += csr->valori[j] * vec->vettore[csr->jValori[j]];
        }
        result->vettore[i] = sum;
    }
}


#define PADDING 32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int coaliscanceMatCsr(MatriceCsr * normale, MatriceCsr **sistemata) {
    if (normale == NULL || sistemata == NULL) {
        fprintf(stderr, "Errore: Puntatori di input non validi.\n");
        return -1;
    }
    *sistemata = NULL; 

    unsigned int *paddings = malloc(sizeof(unsigned int) * normale->height);
    if (paddings == NULL) {
        fprintf(stderr, "Errore: Allocazione memoria fallita per paddings.\n");
        return -1;
    }

    unsigned int totalpad = 0;
    for (int i = 0; i < normale->height; i++) {
        unsigned int elements_in_row = normale->iRP[i + 1] - normale->iRP[i];
        unsigned int pad = PADDING - (elements_in_row % PADDING);
        pad = (pad == PADDING) ? 0 : pad; 
        paddings[i] = pad;
        totalpad += pad;
    }

    *sistemata = malloc(sizeof(MatriceCsr));
    if (*sistemata == NULL) {
        fprintf(stderr, "Errore: Allocazione memoria fallita per la struttura sistemata.\n");
        free(paddings); 
        return -1;
    }

    MatriceCsr *sistemataP = *sistemata;
    sistemataP->height = normale->height;
    sistemataP->width = normale->width;
    sistemataP->nz = normale->nz+totalpad; 
    sistemataP->valori = NULL;  
    sistemataP->jValori = NULL;
    sistemataP->iRP = NULL;

    unsigned int total_elements_padded = sistemataP->nz;

    sistemataP->valori = malloc(sizeof(double) * total_elements_padded);
    if (sistemataP->valori == NULL) {
        fprintf(stderr, "Errore: Allocazione memoria fallita per sistemataP->valori.\n");
        free(paddings);
        free(*sistemata); // Free the main struct
        *sistemata = NULL; // Avoid dangling pointer
        return -1;
    }

    sistemataP->jValori = malloc(sizeof(int) * total_elements_padded);
    if (sistemataP->jValori == NULL) {
        fprintf(stderr, "Errore: Allocazione memoria fallita per sistemataP->jValori.\n");
        free(paddings);
        free(sistemataP->valori); // Clean up previous allocation
        free(*sistemata);
        *sistemata = NULL;
        return -1;
    }

    // Allocate iRP for pairs of (start, end_padded) indices
    sistemataP->iRP = malloc(sizeof(int) * (sistemataP->height * 2));
    if (sistemataP->iRP == NULL) {
        fprintf(stderr, "Errore: Allocazione memoria fallita per sistemataP->iRP.\n");
        free(paddings);
        free(sistemataP->valori);
        free(sistemataP->jValori);
        free(*sistemata);
        *sistemata = NULL;
        return -1;
    }


    unsigned int current_pos = 0;
    for (int i = 0; i < normale->height; i++) {
        unsigned int elements = normale->iRP[i + 1] - normale->iRP[i];
        unsigned int pad = paddings[i];
        unsigned int baseNormale = normale->iRP[i];

        sistemataP->iRP[i * 2] = current_pos; // Start index for row i

        // Copy existing values and column indices
        if (elements > 0) {
             memcpy(&(sistemataP->valori[current_pos]), &(normale->valori[baseNormale]), sizeof(double) * elements);
             memcpy(&(sistemataP->jValori[current_pos]), &(normale->jValori[baseNormale]), sizeof(int) * elements);
        }

        current_pos += elements;

        // Add padding
        if (pad > 0) {
          int last_col_index = (elements > 0) ? normale->jValori[baseNormale + elements - 1] : -1; // Or 0, depending on convention
            for (int p = 0; p < pad; p++) {
                sistemataP->jValori[current_pos] = last_col_index; // Pad with last valid column index
                sistemataP->valori[current_pos] = 0.0;           // Pad with zero value
                current_pos+=1;
            }
        }
        sistemataP->iRP[i * 2 + 1] = current_pos; // End index (exclusive) for padded row i
    }

    free(paddings); // Free the temporary padding array
    return 0; // Success
}

int csrMultWithTime(int (*multiplayer)(struct MatriceCsr *, struct Vector *, struct Vector *), struct MatriceCsr *csr, struct Vector *vec, struct Vector *result, double *execTime)
{
    double t;
    t = omp_get_wtime();
    multiplayer(csr, vec, result);
    t = omp_get_wtime() - t;
    (*execTime) = t; // in seconds
}

int freeMatCsr(struct MatriceCsr **matricePointer)
{
    free((*matricePointer)->iRP);
    free((*matricePointer)->jValori);
    free((*matricePointer)->valori);
    free(*matricePointer);
}
