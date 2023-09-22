cmake_minimum_required(VERSION 3.21)

option(USE_OPENMP "Use OpenMP - by default ON" ON)

set(PCA_TESTS "PcaTests")
PROJECT(${PCA_TESTS})

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- /Zc:__cplusplus")
endif(MSVC)

find_package(OpenMP)
find_package(Catch2 3.3 CONFIG REQUIRED)
find_package(nlohmann_json 3.11 CONFIG REQUIRED)

set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen")

IF(EXISTS "${Eigen3_DIR}/CMakeLists.txt")
    MESSAGE(STATUS "Eigen3 found at ${Eigen3_DIR}")
ELSE()
	MESSAGE(FATAL_ERROR "Eigen3 not found")
ENDIF()

set(PCA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(SOURCES
    test.cpp
    utils.hpp
)

source_group(Tests FILES ${SOURCES})

add_executable(${PCA_TESTS} ${SOURCES})

target_include_directories(${PCA_TESTS} PRIVATE ${PCA_DIR})
target_include_directories(${PCA_TESTS} PRIVATE ${Eigen3_DIR})

target_link_libraries(${PCA_TESTS} PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(${PCA_TESTS} PRIVATE Catch2::Catch2WithMain)

if(USE_OPENMP AND OpenMP_CXX_FOUND)
    message(STATUS "Link to OpenMP")
    target_link_libraries(${PCA_TESTS} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(${PCA_TESTS} PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(${PCA_TESTS} PRIVATE /bigobj)    # for Eigen
endif(MSVC)
