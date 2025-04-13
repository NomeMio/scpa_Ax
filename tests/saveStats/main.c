#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matriciOpp.h"
#include "stats.h"

#define MAX_LINE_LENGTH 256
#define MAX_MATRICES 30 // Define a maximum number of matrices

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s [martix-market-filename] [number of max threads] [number of measure for combination] [lenght of hll blocks]\n", argv[0]);
        exit(1);
    }

    int threads = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    int hack = atoi(argv[4]);

    FILE *pipe;
    char line[MAX_LINE_LENGTH];
    char **matrix_names = NULL;
    int num_matrices = 0;
    const char *command = "../../../addMatrices.sh"; // Adjust path if needed

    pipe = popen(command, "r");
    if (pipe == NULL) {
        perror("Error executing script");
        return 1;
    }

    while (fgets(line, sizeof(line), pipe) != NULL) {
        // Remove trailing newline character
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }

        // Remove leading/trailing whitespace (optional)
        char *start = line;
        while (*start == ' ' || *start == '\t')
            start++;
        char *end = start + strlen(start) - 1;
        while (end > start && (*end == ' ' || *end == '\t'))
            *end-- = '\0';

        // Skip empty lines
        if (*start != '\0') {
            char *name = strdup(start);
            if (!name) {
                perror("strdup");
                /* cleanup */;
                return 1;
            }
            char **temp = realloc(matrix_names, (num_matrices + 1) * sizeof(char *));
            if (!temp) {
                perror("realloc");
                free(name);
                /* cleanup */;
                return 1;
            }
            matrix_names = temp;
            matrix_names[num_matrices++] = name;
        }
    }

    pclose(pipe);

   /*
   for (int i = 0; i < num_matrices; i++) {
        printf("Matrix %d: %s\n", i + 1, matrix_names[i]);
    }

   */
    
    // Allocate results array for all matrices
    struct CsvEntry *all_results = malloc(sizeof(struct CsvEntry) * 4 * num_matrices);
    if (!all_results) {
        perror("Failed to allocate memory for all_results");
        for (int i = 0; i < num_matrices; i++)
            free(matrix_names[i]);
        free(matrix_names);
        return 1;
    }

    struct Vector *vectorR;
    int seed = 42;

    omp_set_num_threads(threads);

    for (int current_matrix = 0; current_matrix < num_matrices; current_matrix++) {
        char full_matrix_path[256]; // Adjust size as needed
        snprintf(full_matrix_path, sizeof(full_matrix_path), "%s/%s", "mat", matrix_names[current_matrix]);

        


        struct MatriceRaw *mat;
        int result = loadMatRaw(full_matrix_path, &mat);
        if (result != 1) {
            fprintf(stderr, "Errore leggendo la matrice: %s\n", full_matrix_path);
            // Cleanup and continue or exit, depending on your error handling policy
            for (int i = 0; i < current_matrix; i++) {
                free(matrix_names[i]);
            }
            free(matrix_names);
            free(all_results);
            return 1; // Or handle the error and continue
        }

        

        struct MatriceCsr *csrMatrice;
        convertRawToCsr(mat, &csrMatrice);
        unsigned int rows = mat->height;

        if (generate_random_vector(seed, rows, &vectorR) != 0) {
            fprintf(stderr, "Failed to allocate memory or invalid input for matrix: %s\n", full_matrix_path);
            freeMatRaw(&mat);
            freeMatCsr(&csrMatrice);
            // Cleanup and continue or exit
            for (int i = 0; i < current_matrix; i++) {
                free(matrix_names[i]);
            }
            free(matrix_names);
            free(all_results);
            return 1;
        }

        

        // Calculate offsets for results array
        int csr_serial_offset = current_matrix * 4 + 0;
        int csr_parallel_offset = current_matrix * 4 + 1;
        int hll_serial_offset = current_matrix * 4 + 2;
        int hll_parallel_offset = current_matrix * 4 + 3;

        struct Vector *resultV1;
        generateEmpty(rows, &resultV1);
        initializeCsvEntry(&all_results[csr_serial_offset], matrix_names[current_matrix], "csr", "serial", 1, 0, iterations);

        

        double time = 0;
        for (int i = 0; i < iterations; i++) {
            csrMultWithTime(&serialCsrMult, csrMatrice, vectorR, resultV1, &time);
            all_results[csr_serial_offset].measure[i] = 2.0 * mat->nz / (time * 1000000000);
        }
        freeRandom(&resultV1);

       

        struct Vector *resultV2;
        generateEmpty(rows, &resultV2);
        initializeCsvEntry(&all_results[csr_parallel_offset], matrix_names[current_matrix], "csr", "parallelOpenMp", threads, 0, iterations);
        time = 0;
        for (int i = 0; i < iterations; i++) {
            csrMultWithTime(&parallelCsrMult, csrMatrice, vectorR, resultV2, &time);
            all_results[csr_parallel_offset].measure[i] = 2.0 * mat->nz / (time * 1000000000);
        }
        freeRandom(&resultV2);

        

        struct MatriceHLL *matHll;
        hack=mat->nz/20;

        convertRawToHll(mat, hack, &matHll);
        struct Vector *resultV3;
        generateEmpty(rows, &resultV3);
        initializeCsvEntry(&all_results[hll_serial_offset], matrix_names[current_matrix], "hll", "serial", 1, hack, iterations);
        time = 0;
        for (int i = 0; i < iterations; i++) {
            hllMultWithTime(&serialMultiplyHLL, matHll, vectorR, resultV3, &time);
            all_results[hll_serial_offset].measure[i] = 2.0 * mat->nz / (time * 1000000000);
        }
        freeRandom(&resultV3);

        

        struct Vector *resultV4;
        generateEmpty(rows, &resultV4);
        initializeCsvEntry(&all_results[hll_parallel_offset], matrix_names[current_matrix], "hll", "parallelOpenMp", threads, hack, iterations);
        time = 0;
        for (int i = 0; i < iterations; i++) {
            hllMultWithTime(&openMpMultiplyHLL, matHll, vectorR, resultV4, &time);
            all_results[hll_parallel_offset].measure[i] = 2.0 * mat->nz / (time * 1000000000);
        }
        freeRandom(&resultV4);

        

        freeRandom(&vectorR);
        freeMatHll(&matHll);
        freeMatRaw(&mat);
        freeMatCsr(&csrMatrice);

        
    }


  

    writeCsvEntriesToFile("../../../result/test.csv", all_results, 4 * num_matrices);

   

    // Free matrix names
    for (int i = 0; i < num_matrices; i++) {
        free(matrix_names[i]);
    }
    free(matrix_names);
    free(all_results);

    return 0;
}

