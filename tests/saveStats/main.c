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

#define MAX_LINE_LENGTH 256
#define RESULTS_PER_MATRIX 4 
#define MATRIX_DIR "mat"
#define DEFAULT_SEED 42
#define SCRIPT_COMMAND "../../../addMatrices.sh" 
#define CSV_OUTPUT_FILE "../../../result/test.csv" 


typedef struct {
    int *thread_counts;
    int num_thread_counts;
    int num_iterations;
    char *single_matrix_file;
} AppConfig;

typedef struct {
    char **names;
    int count;
} MatrixFileList;
    

int parse_arguments(int argc, char *argv[], AppConfig *config);
void print_app_config(const AppConfig *config);
void cleanup_app_config(AppConfig *config);
int get_matrix_filenames(const char *script_path, MatrixFileList *file_list);
int process_matrix(const char *matrix_name ,const AppConfig *config,FILE * csv);
int get_matrix_filenames(const char *script_path, MatrixFileList *file_list);
void cleanup_matrix_filenames(MatrixFileList *file_list);
void print_matrix_file_list(const MatrixFileList *file_list);

int main(int argc, char *argv[]) {
    
    AppConfig config;
    FILE * csv;
    MatrixFileList file_list;

    if (parse_arguments(argc, argv, &config) != 0) {
        return 1;
    } 
    print_app_config(&config);      //Troppo bello pero non l'ho fatta i
    
   
    if(config.single_matrix_file==NULL){
        if (get_matrix_filenames(SCRIPT_COMMAND, &file_list) != 0) {
            return 1;
        }
        print_matrix_file_list(&file_list);

    }else{
        file_list.names=malloc(sizeof(char *));
        (*file_list.names)=malloc(strlen(config.single_matrix_file)*sizeof(char));
        memcpy(*file_list.names,config.single_matrix_file,strlen(config.single_matrix_file)*sizeof(char));
        //*file_list.names=config.single_matrix_file;
        file_list.count=1;
    }
    csv=initialize_csv_file(CSV_OUTPUT_FILE);
    if (csv==NULL){
        return 1;
    }
    
    
    
    for(int i=0;i<file_list.count;i++){
        printf("Processing matrix:%s\n",file_list.names[i]);
        process_matrix(file_list.names[i],&config,csv);
    }

    cleanup:
        close(csv);
        cleanup_matrix_filenames(&file_list);
        cleanup_app_config(&config);
}


int calculateHackSize2(MatriceRaw *rawMat) {
    int R = rawMat->width;
    int nz = rawMat->nz;
    if (nz == 0) return 1;

    // Conta non-zero per riga
    int *row_nz = calloc(R, sizeof(int));
    if (!row_nz) {
        perror("calloc row_nz");
        return R;
    }
    for (int k = 0; k < nz; k++) {
        row_nz[ rawMat->iVettore[k] ]++;
    }

    // Calcola media nz per riga
    double avg = (double)nz / R;
    free(row_nz);

    // Stima footprint per riga (dato + indice)
    double footprint_per_row = avg * (sizeof(double) + sizeof(int));

    // Legge dimensione cache L1
    long cache_L1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    printf("cache L1 size: %ld\n", cache_L1);
    //if (cache_L1 < 0) cache_L1 = 32768;  // fallback a 32KB se sconosciuta

    // Calcola righe che stanno nella cache
    int rows_in_cache = (int)(cache_L1 / footprint_per_row);

    // Clamp hack tra 1 e R
    int hack = rows_in_cache < 1 ? 1 : (rows_in_cache > R ? R : rows_in_cache);

    return hack;
}


int calculateHackSize(MatriceRaw *rawMat) {
    int totalRows = rawMat->width;
    int nz = rawMat->nz;
    if (nz == 0) return 1;

    int *row_nz_counts = calloc(totalRows, sizeof(int));
    if (!row_nz_counts) {
        perror("Errore allocazione row_nz_counts");
        return totalRows;
    }

    for (int i = 0; i < nz; i++) {
        row_nz_counts[rawMat->iVettore[i]]++;
    }

    double avg_nz_per_row = (double)nz / totalRows;
    double var_nz_per_row = 0.0;
    for (int i = 0; i < totalRows; i++) {
        var_nz_per_row += pow(row_nz_counts[i] - avg_nz_per_row, 2);
    }
    var_nz_per_row /= totalRows;

    free(row_nz_counts);
    printf(" varianza: %lf\n",var_nz_per_row);

    // Logica per determinare hack_size in base a avg e var
    int hack_size;
    if (var_nz_per_row < 10) { // Bassa varianza: righe simili
        hack_size = totalRows; // Un unico blocco
    } else if (var_nz_per_row < 100) {
        hack_size = nz / 50; // Valore intermedio
    } else {
        hack_size = nz / 2000; // Alta varianza: blocchi piccoli
    }
    return fmax(1, fmin(hack_size, totalRows)); // Assicurati che sia valido
}

int process_matrix(const char *matrix_name, const AppConfig *config,FILE * csv){
    char full_matrix_path[512]; // Increased buffer size
    snprintf(full_matrix_path, sizeof(full_matrix_path), "../../../%s/%s", MATRIX_DIR, matrix_name);
    int status=-1;
    struct MatriceRaw *mat = NULL;
    struct MatriceCsr *csrMatrice = NULL;
    struct MatriceHLL *matHll = NULL;
    struct Vector *vectorR=NULL;
    if (loadMatRaw(full_matrix_path, &mat) != 1) {
        fprintf(stderr, "Error reading matrix: %s\n", full_matrix_path);
        goto cleanup; // Use goto for centralized cleanup within this function
    }

    int rows=mat->height;
    int seed=1;

    if (generate_random_vector(seed, rows, &vectorR) != 0) {
        fprintf(stderr, "Failed to generate random vector for matrix: %s\n", matrix_name);
        goto cleanup;
    }


    convertRawToCsr(mat, &csrMatrice);
    if (!csrMatrice) {
         fprintf(stderr, "Error converting %s to CSR\n", matrix_name);
         goto cleanup;
    }
    int hack=calculateHackSize2(mat);
    //int hack=mat->nz/200;
    convertRawToHll(mat, hack, &matHll);
     if (!matHll) {
          fprintf(stderr, "Error converting %s to HLL (block length %d)\n", matrix_name, hack);
          goto cleanup;
     }

    double non_zeros = (double)mat->nz; 
    int iterations=config->num_iterations;

    //--------------------------PUT ALL MATRIX OPERATIONS-------------------//
    //------------------------------Serial CSR-----------------------------//

    struct Vector *resultSerial;
    {   
        struct CsvEntry result;
        generateEmpty(rows, &resultSerial);
        initializeCsvEntry(&result, matrix_name, "csr", "serial","simple_linear", mat->nz, 0, 1,0,iterations,0.0,0.0);

        double time = 0;
        for (int i = 0; i < iterations; i++) {
            csrMultWithTime(&serialCsrMult, csrMatrice, vectorR, resultSerial, &time);
            result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
        }
        append_csv_entry(csv,&result);
        //freeRandom(&resultV1);
    }
    //------------------------------OpenMP CSR-----------------------------//

    for(int j=0;j<config->num_thread_counts;j++){
        int thread=config->thread_counts[j];
        omp_set_num_threads(thread);
        struct CsvEntry result;
        struct Vector *resultV;
        generateEmpty(rows, &resultV);
        initializeCsvEntry(&result, matrix_name, "csr", "openMp","simple",mat->nz,0,thread, 0, iterations,0.0,0.0);
        double time = 0;
        for (int i = 0; i < iterations; i++) {
            csrMultWithTime(&parallelCsrMult, csrMatrice, vectorR, resultV, &time);
            result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
        }
        if(areVectorsEqual(resultV,resultSerial)!=0){
            printf("result openMP csr serial is borken");
        }else{
            double diff;
            double percentage;
            calculate_vector_differences(resultV,resultSerial,&diff,&percentage);
            result.errorPercentage=percentage;
            result.errorValue=diff;
            append_csv_entry(csv,&result);
        }
        freeRandom(&resultV);
    }
    //------------------------------Serial HLL-----------------------------//
    {   
        struct Vector *resultV;
        struct CsvEntry result;
        generateEmpty(rows, &resultV);
        initializeCsvEntry(&result, matrix_name, "hll", "serial","simple",mat->nz, hack,1,0 , iterations,0.0,0.0);

        double time = 0;
        for (int i = 0; i < iterations; i++) {
            hllMultWithTime(&serialMultiplyHLL, matHll, vectorR, resultV, &time);
            result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
        }
        if(areVectorsEqual(resultV,resultSerial)!=0){
            printf("result hll serial is borken");
        }else{
           
            append_csv_entry(csv,&result);
        }
        freeRandom(&resultV);
    }
    //------------------------------OpenMP HLL-----------------------------//

    for(int j=0;j<config->num_thread_counts;j++){
        int thread=config->thread_counts[j];
        omp_set_num_threads(thread);
        struct CsvEntry result;
        struct Vector *resultV;
        generateEmpty(rows, &resultV);
        initializeCsvEntry(&result, matrix_name, "hll", "openMp","simple",mat->nz, hack,thread, 0, iterations,0.0,0.0);

        double time = 0;
        for (int i = 0; i < iterations; i++) {
            hllMultWithTime(&openMpMultiplyHLL, matHll, vectorR, resultV, &time);
            result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
        }
        if(areVectorsEqual(resultV,resultSerial)!=0){
            printf("result hll openMP is borken");
        }else{
            double diff;
            double percentage;
            calculate_vector_differences(resultV,resultSerial,&diff,&percentage);
            result.errorPercentage=percentage;
            result.errorValue=diff;
            append_csv_entry(csv,&result);
        }
        freeRandom(&resultV);
    }

//------------------------------CUDA CSR  SERIAL KERNEL-----------------------------//
for(unsigned int  j=32;j<257;j=j*2){
    struct CsvEntry result;
    struct Vector *resultV;
    generateEmpty(rows, &resultV);
    initializeCsvEntry(&result, matrix_name, "csr", "cuda","simple_serial",mat->nz, 0, mat->height,j ,iterations,0.0,0.0);

    double time = 0;
    for (int i = 0; i < iterations; i++) {
        multCudaCSRKernelLinear( csrMatrice, vectorR, resultV, &time,j);
        result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }
    if(areVectorsEqual(resultV,resultSerial)!=0){
        printf("result cuda serial csr is borken");
    }else{
        double diff;
        double percentage;
        calculate_vector_differences(resultV,resultSerial,&diff,&percentage);
        result.errorPercentage=percentage;
        result.errorValue=diff;
        append_csv_entry(csv,&result);
    }
    freeRandom(&resultV); 
}
//------------------------------CUDA CSR  WARP KERNEL-----------------------------//
for(unsigned int  j=32;j<257;j=j*2){
    struct CsvEntry result;
    struct Vector *resultV;
    generateEmpty(rows, &resultV);
    initializeCsvEntry(&result, matrix_name, "csr", "cuda","warp",mat->nz, 0, mat->height,j ,iterations,0.0,0.0);

    double time = 0;
    for (int i = 0; i < iterations; i++) {
        multCudaCSRKernelWarp( csrMatrice, vectorR, resultV, &time,j);
        result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }
    if(areVectorsEqual(resultV,resultSerial)!=0){
        printf("result cuda warp is borken");
    }else{
        double diff;
        double percentage;
        calculate_vector_differences(resultV,resultSerial,&diff,&percentage);
        result.errorPercentage=percentage;
        result.errorValue=diff;
        append_csv_entry(csv,&result);
    }
    freeRandom(&resultV); 
}


MatriceCsr *coal;
if( coaliscanceMatCsr(csrMatrice,&coal)==-1){
        printf("error creating coalescent csr \n");
        exit(-1);
};

for(unsigned int  j=32;j<257;j=j*2){


    struct CsvEntry result;
    struct Vector *resultV;
    generateEmpty(rows, &resultV);
    initializeCsvEntry(&result, matrix_name, "csr", "cuda","warp_coalescent",mat->nz, 0, mat->height,j ,iterations,0.0,0.0);

    double time = 0;
    for (int i = 0; i < iterations; i++) {
        multCudaCSRKernelWarpCoal( coal, vectorR, resultV, &time,j);
        result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }
    if(areVectorsEqual(resultV,resultSerial)!=0){
        printf("result cuda warp is borken");
    }else{
        double diff;
        double percentage;
        calculate_vector_differences(resultV,resultSerial,&diff,&percentage);
        result.errorPercentage=percentage;
        result.errorValue=diff;
        append_csv_entry(csv,&result);
    }
    freeRandom(&resultV); 
    
}
freeMatCsr(&coal);
// CONVERT HLL TO FLAT HLL
FlatELLMatrix *cudaHllMat;
int flatHll = convertHLLToFlatELL(&matHll, &cudaHllMat);




if (flatHll != 0){
        printf("Error while converting to flat format result vector\n");
        return flatHll;
}

//------------------------------CUDA HLL  1-----------------------------//
for(unsigned int  j=32;j<257;j=j*2){
    struct CsvEntry result;
    struct Vector *resultV;
    generateEmpty(rows, &resultV);
    initializeCsvEntry(&result, matrix_name, "hll", "cuda","kernel1",mat->nz, hack, mat->height,j ,iterations,0.0,0.0);

    double time = 0;
    for (int i = 0; i < iterations; i++) {
        int result_ = invokeKernel1(vectorR, resultV, cudaHllMat, matHll, hack, &time,j);// lu segmentation fault qui
        if(result_!=0){
           printf("kernel 1 crashed\n");
            exit(1);
        }
        result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }
    if(areVectorsEqual(resultV,resultSerial)!=0){
        printf("result hll  is borken for kernel1 \n");
    }else{
        double diff;
        double percentage;
        calculate_vector_differences(resultV,resultSerial,&diff,&percentage);
        
        append_csv_entry(csv,&result);
    }
    freeRandom(&resultV); 
}
//------------------------------CUDA HLL  2-----------------------------//
for(unsigned int  j=32;j<129;j=j*2){
    struct CsvEntry result;
    struct Vector *resultV;
    generateEmpty(rows, &resultV);
    initializeCsvEntry(&result, matrix_name, "hll", "cuda","kernel2",mat->nz, hack, mat->height,j ,iterations,0.0,0.0);

    double time = 0;
    for (int i = 0; i < iterations; i++) {
        int booo = invokeKernel2(vectorR, resultV, cudaHllMat, matHll, hack, &time,j);// lu segmentation fault qui
        if(booo!=0){
           printf("kernel 1 crashed\n");
            exit(1);
        }
        result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }
    if(areVectorsEqual(resultV,resultSerial)!=0){
        printf("result hll  is borken for kernel2 \n");
    }else{
        double diff;
        double percentage;
        calculate_vector_differences(resultV,resultSerial,&diff,&percentage);
        
        append_csv_entry(csv,&result);
    }
    freeRandom(&resultV); 
}
//------------------------------CUDA HLL  3-----------------------------//
for(unsigned int  j=32;j<257;j=j*2){
    struct CsvEntry result;
    struct Vector *resultV;
    generateEmpty(rows, &resultV);
    initializeCsvEntry(&result, matrix_name, "hll", "cuda","kernel3",mat->nz, hack, mat->height,j ,iterations,0.0,0.0);

    double time = 0;
    for (int i = 0; i < iterations; i++) {
        int result_ = invokeKernel3(vectorR, resultV, cudaHllMat, matHll, hack, &time,j);// lu segmentation fault qui
        if(result_!=0){
           printf("kernel 1 crashed\n");
            exit(1);
        }
        result.measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }
    if(areVectorsEqual(resultV,resultSerial)!=0){
        printf("result hll  is borken for kernel3 \n");
    }else{
        double diff;
        double percentage;
        calculate_vector_differences(resultV,resultSerial,&diff,&percentage);
        
        append_csv_entry(csv,&result);
    }
    freeRandom(&resultV); 
}


cleanup:
    freeMatHll(&matHll);   // Safe if matHll is NULL
    freeMatCsr(&csrMatrice);// Safe if csrMatrice is NULL
    freeMatRaw(&mat);   
return status;
}






void print_app_config(const AppConfig *config) {
    // Check if the pointer passed is valid
    if (config == NULL) {
        printf("AppConfig pointer is NULL.\n");
        return;
    }

    printf("--- Application Configuration ---\n");

    // Print num_iterations
    printf("Number of Iterations: %d\n", config->num_iterations);

    // Print the list of thread counts
    printf("Thread Counts (%d): ", config->num_thread_counts);
    if (config->thread_counts != NULL && config->num_thread_counts > 0) {
        for (int i = 0; i < config->num_thread_counts; ++i) {
            printf("%d", config->thread_counts[i]);
            // Add a comma and space unless it's the last element
            if (i < config->num_thread_counts - 1) {
                printf(", ");
            }
        }
        printf("\n");
    } else {
        printf("[No thread counts specified or list is empty]\n");
    }

    // Print the optional single matrix file name
    printf("Single Matrix File: ");
    if (config->single_matrix_file != NULL) {
        printf("%s\n", config->single_matrix_file);
    } else {
        printf("[Not specified - will use script]\n");
    }

    printf("-------------------------------\n");
}

int parse_int_list(const char *str, int **list_ptr, int *count_ptr) {
    *list_ptr = NULL;
    *count_ptr = 0;
    if (!str || *str == '\0') {
        fprintf(stderr, "Error: Input string for integer list is NULL or empty.\n");
        return -1;
    }

    char *str_copy = strdup(str);
    if (!str_copy) {
        perror("Error duplicating string for parsing");
        return -1;
    }

    int capacity = 4;
    int count = 0;
    int *list = malloc(capacity * sizeof(int));
    if (!list) {
        perror("Error allocating initial memory for integer list");
        free(str_copy);
        return -1;
    }

    char *token = strtok(str_copy, ",");
    while (token != NULL) {

        while (*token == ' ' || *token == '\t') token++;
        char *end_token = token + strlen(token) - 1;
        while (end_token > token && (*end_token == ' ' || *end_token == '\t')) *end_token-- = '\0';

         if (*token == '\0') {
             token = strtok(NULL, ",");
             continue;
        }

        char *endptr;
        errno = 0;
        long val = strtol(token, &endptr, 10);


        if (endptr == token) { fprintf(stderr, "Error: Invalid number format '%s' in list.\n", token); goto error_cleanup; }
        if (*endptr != '\0') { fprintf(stderr, "Error: Trailing characters after number '%s' in list.\n", token); goto error_cleanup; }
        if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0)) { fprintf(stderr, "Error: Number '%s' out of range for long.\n", token); goto error_cleanup; }
        if (val <= 0 || val > INT_MAX) { fprintf(stderr, "Error: Parsed value '%s' must be a positive integer within range.\n", token); goto error_cleanup; }


        if (count >= capacity) {
            capacity *= 2;
            int *temp = realloc(list, capacity * sizeof(int));
            if (!temp) { perror("Error reallocating memory for integer list"); goto error_cleanup; }
            list = temp;
        }

        list[count++] = (int)val;
        token = strtok(NULL, ",");
    }

    free(str_copy);
    if (count == 0) { fprintf(stderr, "Error: No valid integers found in the list string.\n"); free(list); return -1; }

    *list_ptr = list;
    *count_ptr = count;
    return 0;

error_cleanup:
    free(list);
    free(str_copy);
    *list_ptr = NULL;
    *count_ptr = 0;
    return -1;
}

int parse_arguments(int argc, char *argv[], AppConfig *config) {

    config->thread_counts = NULL;
    config->num_thread_counts = 0;
    config->num_iterations = 0;
    config->single_matrix_file = NULL;


    if (argc < 3) {
        fprintf(stderr, "Usage: %s \"thread_list\" iterations [optional_matrix_file]\n", argv[0]);
        fprintf(stderr, "Example: %s \"1,2,4,8\" 10\n", argv[0]);
        fprintf(stderr, "Example: %s \"1,8\" 5 my_matrix.mtx\n", argv[0]);
        return -1;
    }


    if (parse_int_list(argv[1], &config->thread_counts, &config->num_thread_counts) != 0) {
        fprintf(stderr, "Error parsing thread list: '%s'\n", argv[1]);
        return -1;
    }


    char *endptr;
    errno = 0;
    long iterations_val = strtol(argv[2], &endptr, 10);
    if (endptr == argv[2] || *endptr != '\0' || errno == ERANGE || iterations_val <= 0 || iterations_val > INT_MAX) {
         fprintf(stderr, "Error: Invalid iterations value '%s'. Must be a positive integer.\n", argv[2]);
         cleanup_app_config(config);
         return -1;
    }
    config->num_iterations = (int)iterations_val;


    if (argc >= 4) {
        config->single_matrix_file = strdup(argv[3]);
        if (!config->single_matrix_file) {
            perror("Error duplicating single matrix filename");
            cleanup_app_config(config);
            return -1;
        }
        printf("Info: Optional single matrix file specified: %s\n", config->single_matrix_file);
    } else {
        config->single_matrix_file = NULL;
        printf("Info: No single matrix file specified, will use script execution.\n");
    }

    return 0;
}

void cleanup_app_config(AppConfig *config) {
    if (!config) return;
    free(config->thread_counts);
    config->thread_counts = NULL;
    config->num_thread_counts = 0;

    free(config->single_matrix_file);
    config->single_matrix_file = NULL;
}

// --- Get Matrix Filenames ---
int get_matrix_filenames(const char *script_path, MatrixFileList *file_list) {
    FILE *pipe;
    char line[MAX_LINE_LENGTH];
    file_list->names = NULL;
    file_list->count = 0;

    pipe = popen(script_path, "r");
    if (pipe == NULL) {
        perror("Error executing script");
        return -1;
    }

    while (fgets(line, sizeof(line), pipe) != NULL) {
        // Remove trailing newline
        line[strcspn(line, "\n")] = 0;

        // Basic trim leading/trailing whitespace (simplified)
        char *start = line;
        while (*start && (*start == ' ' || *start == '\t')) start++;
        char *end = start + strlen(start);
        while (end > start && (*(end - 1) == ' ' || *(end - 1) == '\t')) end--;
        *end = '\0';

        if (*start == '\0') continue; // Skip empty lines

        char *name = strdup(start);
        if (!name) {
            perror("strdup failed");
            pclose(pipe);
            cleanup_matrix_filenames(file_list); // Clean up what we have so far
            return -1;
        }

        char **temp = realloc(file_list->names, (file_list->count + 1) * sizeof(char *));
        if (!temp) {
            perror("realloc failed");
            free(name);
            pclose(pipe);
            cleanup_matrix_filenames(file_list);
            return -1;
        }
        file_list->names = temp;
        file_list->names[file_list->count++] = name;
    }

    int status = pclose(pipe);
     if (status == -1) {
          perror("pclose failed");
          // Decide if this is critical, cleanup might still be needed
          return -1;
     } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
          fprintf(stderr, "Script '%s' exited with status %d\n", script_path, WEXITSTATUS(status));
          // Decide if this is critical
          // return -1; // Optional: treat script error as fatal
     }


    return 0;
}


void cleanup_matrix_filenames(MatrixFileList *file_list) {
    if (file_list && file_list->names) {
        for (int i = 0; i < file_list->count; i++) {
            free(file_list->names[i]); // Free each duplicated string
        }
        free(file_list->names); // Free the array of pointers
        file_list->names = NULL;
        file_list->count = 0;
    }
}


void print_matrix_file_list(const MatrixFileList *file_list) {
    if (file_list == NULL) {
        printf("MatrixFileList pointer is NULL.\n");
        return;
    }

    printf("--- Matrix File List ---\n");
    printf("Number of matrices: %d\n", file_list->count);

    if (file_list->names != NULL && file_list->count > 0) {
        printf("Matrix Names:\n");
        for (int i = 0; i < file_list->count; ++i) {
            if (file_list->names[i] != NULL) {
                printf("  [%d]: %s\n", i + 1, file_list->names[i]);
            } else {
                printf("  [%d]: [NULL entry]\n", i + 1);
            }
        }
    } else if (file_list->count <= 0) {
         printf("Matrix Names: [List contains zero entries]\n");
    } else {
        printf("Matrix Names: [Names pointer is NULL]\n");
    }

    printf("------------------------\n");
}
