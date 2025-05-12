#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "matriciOpp.h"
#include <time.h>
#include <omp.h>


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <errno.h>


void swap(struct MatriceRaw *matricePointer, unsigned int indexA, unsigned int indexB) {
    // printf("swapped %d %d\n",indexA,indexB); // Optional: for debugging
    unsigned int tempI = matricePointer->iVettore[indexA];
    unsigned int tempJ = matricePointer->jVettore[indexA];
    double tempV = matricePointer->valori[indexA];

    matricePointer->iVettore[indexA] = matricePointer->iVettore[indexB];
    matricePointer->jVettore[indexA] = matricePointer->jVettore[indexB];
    matricePointer->valori[indexA] = matricePointer->valori[indexB];

    matricePointer->iVettore[indexB] = tempI;
    matricePointer->jVettore[indexB] = tempJ;
    matricePointer->valori[indexB] = tempV;
}

int partition(struct MatriceRaw *matricePointer, int low, int high) {
    if (low >= high) {
        return low;
    }

    unsigned int p1 = matricePointer->iVettore[low]; 
    unsigned int p2 = matricePointer->jVettore[low]; 

    int i = low + 1;
    int j = high;    

    while (1) {
       
        while (i <= high && (matricePointer->iVettore[i] < p1 || (matricePointer->iVettore[i] == p1 && matricePointer->jVettore[i] < p2))) {
            i++;
        }
        while (j >= low + 1 && (matricePointer->iVettore[j] > p1 || (matricePointer->iVettore[j] == p1 && matricePointer->jVettore[j] > p2))) {
            j--;
        }

        // If scanners have crossed or met, partitioning is done
        if (i > j) {
            break;
        }
        swap(matricePointer, i, j);
  
        i++;
        j--;
    }

   swap(matricePointer, low, j); 
    return j; 
}

void quickSort(struct MatriceRaw *matricePointer, int low, int high) {
    if (low < high) {
        int pi = partition(matricePointer, low, high);
        quickSort(matricePointer, low, pi - 1);
        quickSort(matricePointer, pi + 1, high);
    }
}

void quickSortInterface(struct MatriceRaw *matricePointer) {
    quickSort(matricePointer,0,matricePointer->nz-1);
}

char *resolve_symlink(const char *path) {
    char *resolved = realpath(path, NULL);
    if (resolved) {
        return resolved;
    }

    // If realpath failed but it's a symlink, resolve manually
    char buf[PATH_MAX];
    ssize_t len = readlink(path, buf, sizeof(buf) - 1);
    if (len == -1) {
        perror("readlink");
        return NULL;
    }

    buf[len] = '\0';

    // If the link is relative, resolve it based on the directory of `path`
    char *dir = strdup(path);
    if (!dir) {
        perror("strdup");
        return NULL;
    }

    char *slash = strrchr(dir, '/');
    if (slash) {
        *(slash + 1) = '\0';  // Keep the trailing slash
    } else {
        strcpy(dir, "./");
    }

    char combined[PATH_MAX];
    if (buf[0] != '/') {
        snprintf(combined, sizeof(combined), "%s%s", dir, buf);
    } else {
        strncpy(combined, buf, sizeof(combined));
    }

    free(dir);

    // Recurse to resolve the next level
    return resolve_symlink(combined);
}


int loadMatRaw(char *filePath, struct MatriceRaw ** matricePointer)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    unsigned long int M, N, nz=0;   
    filePath=resolve_symlink(filePath);
    printf("%s\n",filePath);
    if ((f = fopen(filePath, "r")) == NULL) {
            printf("Non trovo %s\n",filePath);
            return 0;}

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return 0;
    }


    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)) {
        fprintf(stderr, "Errore: supportate solo matrici sparse in formato coordinate.\n");
        fclose(f);
        return 0;
    }



    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return 0;
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)return -2;


    /* reseve memory for matrices */

    // Prima dell'allocazione della memoria

    if (nz <= 0 || nz > (M * N)) {
        printf("Errore: Numero di elementi non zero (nz=%d) non valido\n", nz);
        return 0;
    }

    // Verifica overflow nella moltiplicazione per sizeof
    size_t size_needed = (size_t)nz * sizeof(int);
    if (size_needed / sizeof(int) != (size_t)nz) {
        printf("Errore: Overflow nel calcolo della dimensione della memoria\n");
        return 0;
    }

    ;
    //Inizializzando la struttura
    *matricePointer=malloc(sizeof(struct MatriceRaw));
    struct MatriceRaw *matrice = *matricePointer;
    matrice->height=M;
    matrice->width=N;
    matrice->nz=nz;


    matrice->iVettore = (int *) malloc(nz * sizeof(int));
    matrice->jVettore = (int *) malloc(nz * sizeof(int));
    matrice->valori = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

     if(mm_is_pattern(matcode)){
           for (int i=0; i<nz; i++)
                     {
                         fscanf(f, "%d %d \n", &matrice->iVettore[i], &matrice->jVettore[i]);
                         matrice->iVettore[i]--;  /* adjust from 1-based to 0-based */
                         matrice->jVettore[i]--;
                         matrice->valori[i]=1;

                     }
     }else{
          for (int i=0; i<nz; i++)
             {
                 fscanf(f, "%d %d %lg\n", &matrice->iVettore[i], &matrice->jVettore[i], &matrice->valori[i]);
                 matrice->iVettore[i]--;  /* adjust from 1-based to 0-based */
                 matrice->jVettore[i]--;
             //    printf("Debug 9: Allocazione vettori matrici iVettore,jVettore,valori per %d elementi\n", matrice->iVettore[0]);

             }
     }


    if (f !=stdin) fclose(f);



     //symmetric matrix
     if (mm_is_symmetric(matcode)) {

        // Calcola il nuovo numero di elementi (nz) aggiungendo un duplicato per ogni elemento off-diagonale
        int new_nz = matrice->nz;
        for (int i = 0; i < matrice->nz; i++) {
            if (matrice->iVettore[i] != matrice->jVettore[i])
                new_nz++;
        }
        
       
        unsigned int *new_i = malloc(new_nz * sizeof(unsigned int));
        unsigned int *new_j = malloc(new_nz * sizeof(unsigned int));
        double *new_val = malloc(new_nz * sizeof(double));
        
        if (new_i == NULL || new_j == NULL || new_val == NULL) {
            perror("Errore di allocazione memoria per matrice simmetrica.");
            free(new_i);
            free(new_j);
            free(new_val);
            return 0;
        }
        
        int count = 0;
      
        for (int i = 0; i < matrice->nz; i++) {
            new_i[count] = matrice->iVettore[i];
            new_j[count] = matrice->jVettore[i];
            new_val[count] = matrice->valori[i];
            count++;
            
            // Se l'elemento è off-diagonale, aggiungi anche il corrispettivo simmetrico
            if (matrice->iVettore[i] != matrice->jVettore[i]) {
                new_i[count] = matrice->jVettore[i];
                new_j[count] = matrice->iVettore[i];
                new_val[count] = matrice->valori[i];
                count++;
            }
        }
        
       
        free(matrice->iVettore);
        free(matrice->jVettore);
        free(matrice->valori);
        
     
        matrice->iVettore = new_i;
        matrice->jVettore = new_j;
        matrice->valori = new_val;
        matrice->nz = new_nz;
    }


    /************************/
    /* now write out matrix */
    /************************/

    #ifdef DEBUG 
    printf("ciao")
    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i=0; i<nz; i++)
        fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);
    #endif
    

    
	return 1;
}

int multMatrixAndMeasureTime(int (*multiplayer)(void *,struct Vector *,struct Vector *),void *matrice,struct Vector *vec,struct Vector *result,double *execTime){
    clock_t t;
    t = clock();
    int value=multiplayer(matrice, vec, result);
    t = clock() - t;
    (*execTime) = ((double)t) / CLOCKS_PER_SEC; // in seconds
    return value;
}



int freeMatRaw(struct MatriceRaw ** matricePointer){
    struct MatriceRaw *mat=*matricePointer;
    free(mat->iVettore);
    free(mat->jVettore);
    free(mat->valori);
    free(mat);
    return 0;
}
