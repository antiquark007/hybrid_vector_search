cmake_minimum_required(VERSION 3.16)
project(hybrid_vector_search VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ─── Compiler flags ──────────────────────────────────────────────────────────
add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)

# ─── OpenMP ──────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ─── pybind11 ────────────────────────────────────────────────────────────────
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ─── Include dirs ─────────────────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)

# ─── Python module ───────────────────────────────────────────────────────────
pybind11_add_module(hvs_core src/core/bindings.cpp)
target_link_libraries(hvs_core PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(hvs_core PRIVATE -O3 -march=native -ffast-math)

# ─── Standalone benchmark binary ─────────────────────────────────────────────
add_executable(benchmark src/core/benchmark.cpp)
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(benchmark PRIVATE -O3 -march=native -ffast-math)

# ─── Tests ───────────────────────────────────────────────────────────────────
enable_testing()
add_executable(test_hnsw tests/test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME hnsw_unit COMMAND test_hnsw)
