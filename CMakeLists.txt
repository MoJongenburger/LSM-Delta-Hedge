cmake_minimum_required(VERSION 3.16)
project(LSM_Delta_Hedge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core library
add_library(lsm_core
    src/cpp/lsm/lsm_model.cpp
)
target_include_directories(lsm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Eigen: prefer system Eigen3, fallback to external/eigen (submodule)
find_package(Eigen3 QUIET)
if (Eigen3_FOUND)
    target_link_libraries(lsm_core PUBLIC Eigen3::Eigen)
else()
    message(STATUS "Eigen3 not found via find_package; using external/eigen include path")
    target_include_directories(lsm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)
endif()

# Tests
add_executable(test_smoke tests/cpp/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE lsm_core)

