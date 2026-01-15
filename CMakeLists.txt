cmake_minimum_required(VERSION 3.16)
project(LSM_Delta_Hedge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------
# Python (ABI + install dirs)
# ----------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# ----------------------------
# Core library (C++)
# ----------------------------
add_library(lsm_core
    src/cpp/lsm/lsm_model.cpp
)
target_include_directories(lsm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Eigen: prefer system Eigen3, fallback to external/eigen (submodule or vendored folder)
find_package(Eigen3 QUIET)
if (Eigen3_FOUND)
    target_link_libraries(lsm_core PUBLIC Eigen3::Eigen)
else()
    message(STATUS "Eigen3 not found via find_package; using external/eigen include path")
    target_include_directories(lsm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)
endif()

# ----------------------------
# Tests (C++)
# ----------------------------
add_executable(test_smoke tests/cpp/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE lsm_core)

# ----------------------------
# pybind11: system-first (pip), else submodule fallback
# ----------------------------
option(USE_SYSTEM_PYBIND11 "Use system-installed pybind11 if found" ON)

if (USE_SYSTEM_PYBIND11)
    find_package(pybind11 CONFIG QUIET)
endif()

if (pybind11_FOUND)
    message(STATUS "Using system pybind11 (found via CMake config)")
else()
    message(STATUS "System pybind11 not found; trying submodule external/pybind11")
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11/CMakeLists.txt")
        message(FATAL_ERROR "pybind11 not found. Install pybind11 (pip) or add submodule at external/pybind11.")
    endif()
    add_subdirectory(external/pybind11)
endif()

# ----------------------------
# PYBIND11 MODULE (Python extension)
# ----------------------------
pybind11_add_module(lsm_cpp
    src/cpp/bindings/pybind_module.cpp
)
target_link_libraries(lsm_cpp PRIVATE lsm_core)
target_include_directories(lsm_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Visibility + nicer shared lib behavior
set_target_properties(lsm_cpp PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# RPATH: make editable/dev installs more robust
if(APPLE)
    set_target_properties(lsm_cpp PROPERTIES
        INSTALL_RPATH "@loader_path"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
elseif(UNIX)
    set_target_properties(lsm_cpp PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# ----------------------------
# Install target (optional; CI doesn't need it)
# ----------------------------
if (DEFINED Python3_SITELIB)
    set(PY_SITE_DIR "${Python3_SITELIB}")
elseif(DEFINED Python3_SITEARCH)
    set(PY_SITE_DIR "${Python3_SITEARCH}")
else()
    set(PY_SITE_DIR "python")
endif()

install(TARGETS lsm_cpp
    LIBRARY DESTINATION "${PY_SITE_DIR}/lsmhedge"
    RUNTIME DESTINATION "${PY_SITE_DIR}/lsmhedge"
)
