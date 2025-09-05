cmake_minimum_required(VERSION 3.16)
project(LaplacianChallenge LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directory for binaries
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# Common compiler warnings
set(COMMON_FLAGS -Wall -Wextra -Wpedantic -fopenmp -march=native)
if(DEFINED DEBUG AND DEBUG)
    list(APPEND COMMON_FLAGS -g)
    message(WARNING "Debug mode enabled.")
endif()
# Optimization levels and flags
#set(OPT_LEVELS O1 O2 O3)
set(OPT_LEVELS O3)
set(EXTRA_FLAGS none fastmath unroll vectorize fastmath_unroll_vec)

# Map flags to actual compiler options
set(FLAG_none "")
set(FLAG_fastmath "-ffast-math")
set(FLAG_unroll "-funroll-all-loops")
set(FLAG_vectorize "-ftree-vectorize")
set(FLAG_fastmath_unroll_vec "-ffast-math;-funroll-all-loops;-ftree-vectorize")

# Kokkos configuration
set(KOKKOS_HAS_CPU OFF)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
if(OpenMP_FOUND AND TBB_FOUND)
    set(KOKKOS_HAS_CPU ON)
    message(STATUS "Kokkos CPU support enabled.")
endif()
set(KOKKOS_HAS_GPU OFF)
if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    set(KOKKOS_HAS_GPU ON)
    message(STATUS "Kokkos GPU support enabled.")
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/../../external/kokkos ${CMAKE_BINARY_DIR}/kokkos_build)


foreach(OPT ${OPT_LEVELS})
    foreach(FLAG ${EXTRA_FLAGS})
        set(FLAG_OPTION ${FLAG_${FLAG}})

        if(KOKKOS_HAS_GPU)
            # Kokkos GPU
            add_executable(laplacian_kokkos_gpu_${OPT}_${FLAG} kokkos/run.cpp)

            target_link_libraries(laplacian_kokkos_gpu_${OPT}_${FLAG} PRIVATE OpenMP::OpenMP_CXX)
            target_link_libraries(laplacian_kokkos_gpu_${OPT}_${FLAG} PRIVATE TBB::tbb)
            target_compile_options(laplacian_kokkos_gpu_${OPT}_${FLAG} PRIVATE -${OPT} ${FLAG_OPTION} ${COMMON_FLAGS})
            target_include_directories(laplacian_kokkos_gpu_${OPT}_${FLAG} PRIVATE ${KOKKOS_DIR}/external/kokkos/install_gpu/include)
            target_link_directories(laplacian_kokkos_gpu_${OPT}_${FLAG} PRIVATE ${KOKKOS_DIR}/external/kokkos/install_gpu/lib)
            target_link_libraries(laplacian_kokkos_gpu_${OPT}_${FLAG} PRIVATE kokkoscore kokkosalgorithms kokkoscontainers kokkossimd dl pthread cuda cudart OpenMP::OpenMP_CXX TBB::tbb)
        else()
            # Native, without Kokkos
            add_executable(laplacian_native_${OPT}_${FLAG} native/run.cpp)
            target_compile_options(laplacian_native_${OPT}_${FLAG} PRIVATE -${OPT} ${FLAG_OPTION} ${COMMON_FLAGS})
            target_link_libraries(laplacian_native_${OPT}_${FLAG} PRIVATE OpenMP::OpenMP_CXX)

            if(KOKKOS_HAS_CPU)
                # Kokkos CPU
                add_executable(laplacian_kokkos_cpu_${OPT}_${FLAG} kokkos/run.cpp)

                target_link_libraries(laplacian_kokkos_cpu_${OPT}_${FLAG} PRIVATE OpenMP::OpenMP_CXX)
                target_link_libraries(laplacian_kokkos_cpu_${OPT}_${FLAG} PRIVATE TBB::tbb)
                target_compile_options(laplacian_kokkos_cpu_${OPT}_${FLAG} PRIVATE -${OPT} ${FLAG_OPTION} ${COMMON_FLAGS})
                target_include_directories(laplacian_kokkos_cpu_${OPT}_${FLAG} PRIVATE ${KOKKOS_DIR}/external/kokkos/install_cpu/include)
                target_link_directories(laplacian_kokkos_cpu_${OPT}_${FLAG} PRIVATE ${KOKKOS_DIR}/external/kokkos/install_cpu/lib)
                target_link_libraries(laplacian_kokkos_cpu_${OPT}_${FLAG} PRIVATE kokkoscore kokkosalgorithms kokkoscontainers kokkossimd dl pthread OpenMP::OpenMP_CXX TBB::tbb)
            endif()        
        endif()
    endforeach()
endforeach()

