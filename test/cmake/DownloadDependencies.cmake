include(FetchContent)

set(BUILD_TESTING OFF CACHE BOOL "Enable testing for Eigen" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)
set(EIGEN_BUILD_SPBENCH OFF CACHE BOOL "Enable creation of Eigen benchmarks" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Enable creation of Eigen package config" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "Enable creation of Eigen demos" FORCE)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 5.0.0
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(Eigen3)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2
    GIT_TAG v3.11.0
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nlohmann_json)
