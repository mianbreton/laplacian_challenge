# --- Output directory for test binaries ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/bin)
# --- Common compile flags ---
set(COMMON_FLAGS -g -Wall -Wextra -Wpedantic -O0 -fopenmp)
# --- GoogleTest ---
include(FetchContent)
FetchContent_Declare(
  googletest
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/googletest
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)


# ---------------------------------------------------------------
# Helper function to create tests easily
# ---------------------------------------------------------------
function(add_my_test TARGET_NAME SRC_FILE)
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_FLAGS})
    target_link_libraries(${TARGET_NAME} PRIVATE
        OpenMP::OpenMP_CXX
        gtest_main
    )
    gtest_discover_tests(${TARGET_NAME})
endfunction()

# ---------------------------------------------------------------
# GPU Test
# ---------------------------------------------------------------
if(KOKKOS_HAS_GPU)
    add_my_test(test_kokkos_gpu test_kokkos.cpp)
    target_link_libraries(test_kokkos_gpu PRIVATE
        TBB::tbb
        kokkoscore kokkosalgorithms kokkoscontainers kokkossimd
        dl pthread cuda cudart
    )

# ---------------------------------------------------------------
# CPU / Native Tests
# ---------------------------------------------------------------
else()
    # Native test (no Kokkos)
    add_my_test(test_native test_native.cpp)

    # Kokkos CPU test (if CPU backend available)
    if(KOKKOS_HAS_CPU)
        add_my_test(test_kokkos_cpu test_kokkos.cpp)
        target_link_libraries(test_kokkos_cpu PRIVATE
            TBB::tbb
            kokkoscore kokkosalgorithms kokkoscontainers kokkossimd
            dl pthread
        )
    endif()
endif()