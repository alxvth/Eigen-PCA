include(FetchContent)

set(BUILD_TESTING OFF CACHE BOOL "Enable testing for Eigen" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 3866cbfbe8622f41b4f9fa17227aaa7a8de13890
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(Eigen3)


FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(Catch2)


FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nlohmann_json)