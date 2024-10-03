cmake_minimum_required(VERSION 3.21)

option(USE_OPENMP "Use OpenMP - by default ON" ON)

set(PCA_TESTS "PcaTests")
PROJECT(${PCA_TESTS})

if(MSVC)
    add_compile_options(/permissive- /Zc:__cplusplus)
endif(MSVC)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DownloadDependencies.cmake")

find_package(OpenMP)
find_package(Catch2 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

set(SOURCES
    test.cpp
    utils.hpp
)

source_group(Tests FILES ${SOURCES})

add_executable(${PCA_TESTS} ${SOURCES})

set(PCA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(${PCA_TESTS} PRIVATE ${PCA_DIR})

target_link_libraries(${PCA_TESTS} PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(${PCA_TESTS} PRIVATE Catch2::Catch2WithMain)
target_link_libraries(${PCA_TESTS} PRIVATE Eigen3::Eigen)

if(USE_OPENMP AND OpenMP_CXX_FOUND)
    message(STATUS "Link to OpenMP")
    target_link_libraries(${PCA_TESTS} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(${PCA_TESTS} PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(${PCA_TESTS} PRIVATE /bigobj)    # for Eigen
endif(MSVC)
