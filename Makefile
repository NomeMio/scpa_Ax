.PHONY: run-test-cmake clean

# Directory for building

BUILDS_FOLDER=builds/
TEST_FOLDER=tests/

TEST_FOLDER_CMAKE=$(TEST_FOLDER)cmake
BUILD_DIR_TEST_CMAKE = $(BUILDS_FOLDER)$(TEST_FOLDER_CMAKE)

TEST_FOLDER_MATRICI=$(TEST_FOLDER)open_matrici
BUILD_DIR_TEST_MATRICI=$(BUILDS_FOLDER)$(TEST_FOLDER_MATRICI)

TEST_FOLDER_CSR_MULT=$(TEST_FOLDER)csrMultiplication
BUILD_DIR_TEST_CSR_MULT=$(BUILDS_FOLDER)$(TEST_FOLDER_CSR_MULT)

TEST_OPEN_HLL=$(TEST_FOLDER)open_hll
BUILD_DIR_OPEN_HLL=$(BUILDS_FOLDER)$(TEST_OPEN_HLL)

TEST_OPEN_CUDA=$(TEST_FOLDER)cudaTest
BUILD_DIR_OPEN_CUDA=$(BUILDS_FOLDER)$(TEST_OPEN_CUDA)

TEST_OPEN_CUDAHLL=$(TEST_FOLDER)cudaHll
BUILD_DIR_OPEN_CUDAHLL=$(BUILDS_FOLDER)$(TEST_OPEN_CUDAHLL)

TEST_SAVE_STATS=$(TEST_FOLDER)saveStats
BUILD_DIR_STATS_TEST=$(BUILDS_FOLDER)$(TEST_SAVE_STATS)

TEST_CUDA_HELLO=$(TEST_FOLDER)cudaHello
BUILD_DIR_CUDA_HELLO_TEST=$(BUILDS_FOLDER)$(TEST_CUDA_HELLO)


TEST_FAST=$(TEST_FOLDER)personaleTests
BUILD_DIR_TEST_FAST=$(BUILDS_FOLDER)$(TEST_FAST)


CURRENT_DIR := $(shell pwd)



clean:
	rm -rf $(BUILDS_FOLDER)




build-test-cmake:
	mkdir -p $(BUILD_DIR_TEST_CMAKE)
	cd $(BUILD_DIR_TEST_CMAKE) && cmake $(CURRENT_DIR)/$(TEST_FOLDER_CMAKE)
	cd $(BUILD_DIR_TEST_CMAKE) && cmake --build .
	#cd $(BUILD_DIR_TEST_CMAKE) && ./Main

run-test-cmake:
	cd $(BUILD_DIR_TEST_CMAKE) && ./Main


build-test-matrici:
	mkdir -p $(BUILD_DIR_TEST_MATRICI)
	cd $(BUILD_DIR_TEST_MATRICI) && cmake $(CURRENT_DIR)/$(TEST_FOLDER_MATRICI)
	cd $(BUILD_DIR_TEST_MATRICI) && cmake --build .
	#cd $(BUILD_DIR_TEST_MATRICI) && ./Main



run-test-matrici:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi
	cd $(BUILD_DIR_TEST_MATRICI) && ./Main $(CURRENT_DIR)/$(MATRICE)

	


build-test-csrM:
	mkdir -p $(BUILD_DIR_TEST_CSR_MULT)
	cd $(BUILD_DIR_TEST_CSR_MULT) && cmake $(CURRENT_DIR)/$(TEST_FOLDER_CSR_MULT)
	cd $(BUILD_DIR_TEST_CSR_MULT) && cmake --build .
	#cd $(BUILD_DIR_TEST_CSR_MULT) && ./Main

run-test-csrM:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi
	cd $(BUILD_DIR_TEST_CSR_MULT) && ./Main $(CURRENT_DIR)/$(MATRICE)

build-test-HLL:
	mkdir -p $(BUILD_DIR_OPEN_HLL)
	cd $(BUILD_DIR_OPEN_HLL) && cmake $(CURRENT_DIR)/$(TEST_OPEN_HLL)
	cd $(BUILD_DIR_OPEN_HLL) && cmake --build .
	#cd $(BUILD_DIR_OPEN_HLL) && ./Main



run-test-HLL:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi 
	cd $(BUILD_DIR_OPEN_HLL) && valgrind ./Main $(CURRENT_DIR)/$(MATRICE) $(P)


build-test-stats:
	mkdir -p $(BUILD_DIR_STATS_TEST)
	cd $(BUILD_DIR_STATS_TEST) && cmake $(CURRENT_DIR)/$(TEST_SAVE_STATS)
	cd $(BUILD_DIR_STATS_TEST) && cmake --build .
	#cd $(BUILD_DIR_STATS_TEST) && ./Main



run-test-stats:
	@echo "Checking parameters..."
	@{ \
	    if [ -z "$(ARGS)" ]; then \
	        echo "ERROR: ARGS PATH is not set! Use ARGS='arguments'"; exit 1; \
	    fi; \
	    echo "All parameters are set. Running test..."; \
	    cd $(BUILD_DIR_STATS_TEST) &&  ./Main $(ARGS); \
	}



build-test-cuda-hello:
	mkdir -p $(BUILD_DIR_CUDA_HELLO_TEST)
	cd $(BUILD_DIR_CUDA_HELLO_TEST) && cmake $(CURRENT_DIR)/$(TEST_CUDA_HELLO)
	cd $(BUILD_DIR_CUDA_HELLO_TEST) && cmake --build .
	#cd $(BUILD_DIR_CUDA_HELLO_TEST) && ./Main



run-test-cuda-hello:
	cd $(BUILD_DIR_CUDA_HELLO_TEST) && ./main 

build-test-fast:
	mkdir -p $(BUILD_DIR_TEST_FAST)
	cd $(BUILD_DIR_TEST_FAST) && cmake $(CURRENT_DIR)/$(TEST_FAST)
	cd $(BUILD_DIR_TEST_FAST) && cmake --build .
	#cd $(BUILD_DIR_TEST_FAST) && ./Main



run-test-fast:
	@echo "Checking parameters..."
	@{ \
	    if [ -z "$(MATRICE)" ]; then \
	        echo "ERROR: MATRICE PATH is not set! Use MATRICE=PATH"; exit 1; \
	    fi;\
	    echo "All parameters are set. Running test..."; \
		cd $(BUILD_DIR_TEST_FAST) && ./main $(CURRENT_DIR)/$(MATRICE) ;\
	}


run-test-cuda:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi 
	cd $(BUILD_DIR_OPEN_CUDA) && ./Main $(CURRENT_DIR)/$(MATRICE) $(P)


build-test-cuda:
	mkdir -p $(BUILD_DIR_OPEN_CUDA)
	cd $(BUILD_DIR_OPEN_CUDA) && cmake $(CURRENT_DIR)/$(TEST_OPEN_CUDA)
	cd $(BUILD_DIR_OPEN_CUDA) && cmake --build .
	#cd $(BUILD_DIR_OPEN_CUDA) && ./Main


run-test-cudaHll:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi 
	cd $(BUILD_DIR_OPEN_CUDAHLL) && ./Main $(CURRENT_DIR)/$(MATRICE) $(P) $(K)
	


build-test-cudaHll:
	mkdir -p $(BUILD_DIR_OPEN_CUDAHLL)
	cd $(BUILD_DIR_OPEN_CUDAHLL) &&  cmake $(CURRENT_DIR)/$(TEST_OPEN_CUDAHLL)
	cd $(BUILD_DIR_OPEN_CUDAHLL) && cmake --build .
	#cd $(BUILD_DIR_OPEN_CUDAHLL) && ./Main



build-test-cudaHll2:
	module load cuda/12.8 && \
	mkdir -p $(BUILD_DIR_OPEN_CUDAHLL) && \
	cd $(BUILD_DIR_OPEN_CUDAHLL) && \
	cmake -DCMAKE_CUDA_ARCHITECTURES=75 $(CURRENT_DIR)/$(TEST_OPEN_CUDAHLL) && \
	cmake --build .	








