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
	cd $(BUILD_DIR_TEST_MATRICI) &&  ./Main $(CURRENT_DIR)/$(MATRICE)

	


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
