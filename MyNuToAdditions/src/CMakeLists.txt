project(MyNuToAdditions)
cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 14)

FIND_PACKAGE( OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

find_path(EIGEN_INCLUDE_DIR NAMES Eigen/Core PATH_SUFFIXES eigen3)
include_directories(${EIGEN_INCLUDE_DIR})

include(ExternalProject)

ExternalProject_Add(nuto
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}"
               "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../NuTo"
    BUILD_ALWAYS 1
    )

include_directories(before ${PROJECT_SOURCE_DIR}/../../NuTo)
link_directories("${CMAKE_CURRENT_BINARY_DIR}/lib")

foreach(inputFile
        rectangle10x10.msh
        rectangle100x100.msh
    )
    execute_process(COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink"
        "${CMAKE_CURRENT_SOURCE_DIR}/${inputFile}"
        "${CMAKE_CURRENT_BINARY_DIR}/${inputFile}")

endforeach()

foreach(myFile
    #MeshTopology
    ParallelAssembly
    Coordinates
    ExtendedMeshTest
    createCellsBenchmark
    Plot)
ADD_EXECUTABLE(${myFile} ${myFile}.cpp)
add_dependencies(${myFile} nuto)
TARGET_LINK_LIBRARIES(${myFile} NuTo ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${ARPACK_LIBRARY})
endforeach(myFile)

