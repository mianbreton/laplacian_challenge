set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/tests/bin)
set(COMMON_FLAGS -g -Wall -Wextra -Wpedantic -fopenmp -march=native)

if(KOKKOS_HAS_GPU)
    # Kokkos GPU
    set(EXE_NAME test_kokkos_gpu)
    add_executable(${EXE_NAME} test_kokkos.cpp)
    target_compile_options(${EXE_NAME} PRIVATE -O0 ${COMMON_FLAGS})
    target_link_libraries(${EXE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${EXE_NAME} PRIVATE TBB::tbb)
    target_link_libraries(${EXE_NAME} PRIVATE kokkoscore kokkosalgorithms kokkoscontainers kokkossimd dl pthread cuda cudart OpenMP::OpenMP_CXX TBB::tbb)
    add_test(NAME TEST_KOKKOS_GPU COMMAND ${EXE_NAME})  

else()
    # Native, without Kokkos
    set(EXE_NAME test_native)
    add_executable(${EXE_NAME} test_native.cpp)
    target_compile_options(${EXE_NAME} PRIVATE -O0 ${COMMON_FLAGS})
    target_link_libraries(${EXE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    add_test(NAME TEST_NATIVE COMMAND ${EXE_NAME})
    if(KOKKOS_HAS_CPU)
        # Kokkos CPU
        set(EXE_NAME test_kokkos_cpu)
        add_executable(${EXE_NAME} test_kokkos.cpp)
        target_compile_options(${EXE_NAME} PRIVATE -O0 ${COMMON_FLAGS})

        target_link_libraries(${EXE_NAME} PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(${EXE_NAME} PRIVATE TBB::tbb)
        target_link_libraries(${EXE_NAME} PRIVATE kokkoscore kokkosalgorithms kokkoscontainers kokkossimd dl pthread OpenMP::OpenMP_CXX TBB::tbb)
        add_test(NAME TEST_KOKKOS_CPU COMMAND ${EXE_NAME})  
    endif()
endif()