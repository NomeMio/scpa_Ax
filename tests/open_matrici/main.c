
#define PRINT 0
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include "matriciOpp.h" 
#include "stats.h"     
#include "cuda_alex.h"
#include "cuda_luca.h"
#include <unistd.h>

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }
    struct MatriceRaw *mat;
    int result = loadMatRaw(argv[1], &mat);
    //quickSort(mat,0,mat->nz-1);
    if (result != 1)
    {
        printf("Errore leggendo la matrice");
        return 0;
    }

    fprintf(stdout, "nz=%d height=%d width=%d\n", mat->nz, mat->height, mat->width);
#if PRINT == 1
    for (int i = 0; i < mat->nz; i++)
    {
        fprintf(stdout, "%d %d %20.19g\n", mat->iVettore[i], mat->jVettore[i], mat->valori[i]);
    }
#endif
    //struct MatriceCsr *csrMatrice;
    
    //convertRawToCsr(mat, &csrMatrice); 
    
#if PRINT == 1
    printf("[ ");
    for (int i = 0; i <= csrMatrice->width; i++)
    {
        printf("%d ", csrMatrice->iRP[i]);
    }
    printf("]\n");
    printf("[ ");
    for (int i = 0; i < csrMatrice->nz; i++)
    {
        printf("%d ", csrMatrice->jValori[i]);
    }
    printf("]\n");
    printf("[ ");
    for (int i = 0; i < csrMatrice->nz; i++)
    {
        printf("%f ", csrMatrice->valori[i]);
    }
    printf("]\n");
#endif
struct MatriceHLL *hll;
    int error= convertRawToHll2(mat,1000,&hll);
    if (error!=1){
        printf("error with hll2 inizi  %d\n",error);
        return 0;
    };
printHLL(&hll);
FlatELLMatrix *flat;
convertHLLToFlatELL(&hll,&flat);
printFlatELLMatrix(&flat);
freeMatHll(&hll);

freeMatRaw(&mat);
}
