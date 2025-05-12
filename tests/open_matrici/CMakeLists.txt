cmake_minimum_required(VERSION 3.25)
set(CMAKE_CUDA_ARCHITECTURES 75)
project(openM LANGUAGES C CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA Toolkit
find_package(CUDA REQUIRED)

# Set build type and flags
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fopenmp")

# Add matLib subdirectory
add_subdirectory(../../lib/mat mat_lib_build)

# Apply OpenMP to matlib
if(OpenMP_C_FOUND)
    target_link_libraries(matlib PUBLIC OpenMP::OpenMP_C)
endif()

# Link CUDA runtime to matlib (modern CMake style)
target_link_libraries(matlib PUBLIC cudart)

# Create Main executable
add_executable(Main main.c)
target_compile_options(matlib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

# Link OpenMP and matlib to Main
if(OpenMP_C_FOUND)
    target_link_libraries(Main PUBLIC OpenMP::OpenMP_C)
endif()
target_link_libraries(Main PRIVATE matlib)

# Include matlib headers
target_include_directories(Main PRIVATE ../../lib/mat)
