#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RANDOM_VECTOR_MAX_VALUE 1.0
#define EPSILON 1e-5  // Tolleranza per il confronto tra floating-point

/* Definizione di Vector */
typedef struct Vector {
    unsigned int righe;
    double *vettore;
} Vector;
int calculate_vector_differences(const Vector *vec1, const Vector *vec2, double *sum_abs_diff, double *perc_diff);
int areVectorsEqual( Vector *v1,  Vector *v2);
int generate_random_vector(int seed, unsigned int righe, Vector **pointerToVector);
int freeRandom(Vector **pointerToVector);
int generateEmpty(unsigned int rows, Vector **vettore);
void printVector(Vector *vec);

/* Definizione di MatriceRaw */
typedef struct MatriceRaw { 
    unsigned int width, height;
    unsigned int nz;
    unsigned int *iVettore;
    unsigned int *jVettore;
    double *valori;
} MatriceRaw;
void quickSort(struct MatriceRaw *matricePointer, int low, int high);
void quickSortInterface(struct MatriceRaw *matricePointer);
int loadMatRaw(char *filePath, MatriceRaw **matricePointer);
int freeMatRaw(MatriceRaw **matricePointer);

/* Definizione di MatriceCsr */
typedef struct MatriceCsr {
    unsigned int width, height;
    unsigned int nz; 
    unsigned int *iRP;      // Array di dimensione height+1
    unsigned int *jValori;  // Array di dimensione nz
    double *valori;         // Array di dimensione nz
} MatriceCsr;
int coaliscanceMatCsr(MatriceCsr * normale, MatriceCsr **sistemata);
int parallelCsrMult(MatriceCsr *csr, Vector *vec, Vector *result);
int serialCsrMult(MatriceCsr *csr, Vector *vec, Vector *result);
int csrMultWithTime(int (*multiplayer)(MatriceCsr *, Vector *, Vector *),
                    MatriceCsr *csr, Vector *vec, Vector *result, double *execTime);
int convertRawToCsr(MatriceRaw *matricePointer, MatriceCsr **csrPointer);
int freeMatCsr(MatriceCsr **matricePointer);

/* Definizione di ELLPACK_Block */
typedef struct ELLPACK_Block{
    int M;      // Righe del blocco (HackSize o meno)
    int N;      // Colonne del blocco
    int MAXNZ;  // Numero massimo di elementi non zero per riga
    int *JA;    // Indici di colonna
    double *AS; // Valori dei coefficienti
} ELLPACK_Block;

/* Definizione di MatriceHLL */
typedef struct MatriceHLL {
    int totalRows;          // Numero totale di righe della matrice globale
    int totalCols;          // Numero totale di colonne della matrice globale
    int HackSize;           // Numero di righe per blocco
    int numBlocks;          // Numero di blocchi totali
    ELLPACK_Block **blocks; // Array di blocchi
} MatriceHLL;

/* Definizione di Flat ELLPACK Matrix */
typedef struct FlatELLMatrix {
    double* values_flat;       // Array dei valori flattenati
    int*    col_indices_flat;  // Array degli indici di colonna flattenati
    int*    block_offsets;     // Offset di inizio di ogni blocco
    int*    block_nnz;         // Numero massimo di non zero per riga per ogni blocco (MAXNZ)
    int*    block_rows;        // Numero di righe effettive per ogni blocco
    int hack;
    int     total_values;      // Numero totale di elementi (lunghezza degli array flat)
    int     numBlocks;         // Numero di blocchi
} FlatELLMatrix;
int flatHllMultOpenMP(struct FlatELLMatrix *hll, struct Vector *vec, struct Vector *result, double *execTime);
int convertRawToHll2(struct MatriceRaw *matricePointer, int hackSizeP, struct MatriceHLL **hllP);
int convertRawToEllpack(MatriceRaw *matricePointer, int hackSize, ELLPACK_Block **block);
int convertRawToHll(MatriceRaw *matricePointer, int hackSizeP, MatriceHLL **hll);
int serialMultiplyHLL(MatriceHLL *mat, Vector *vec, Vector *result);
int hllMultWithTime(int (*multiplayer)(MatriceHLL *, Vector *, Vector *),
                    MatriceHLL *hll, Vector *vec, Vector *result, double *execTime);
                    int openMpMultiplyHllFlatFast(struct FlatELLMatrix *mat, struct Vector *vec, struct Vector *result) ;
int openMpMultiplyHLL(MatriceHLL *mat, Vector *vec, Vector *result);
int printHLL(MatriceHLL **hllP);
int freeMatHll(struct MatriceHLL **matricePointer);
#ifdef __cplusplus
}
#endif