project(MyNuToAdditions)
cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 14)

find_path(EIGEN_INCLUDE_DIR NAMES Eigen/Core PATH_SUFFIXES eigen3)
include_directories(${EIGEN_INCLUDE_DIR})
include_directories("nuto")

include(ExternalProject)

ExternalProject_Add(nuto
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}"
               "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../NuTo"
    BUILD_ALWAYS 1
    )

include_directories(${PROJECT_SOURCE_DIR}/../../NuTo)
link_directories("${CMAKE_CURRENT_BINARY_DIR}/lib")

foreach(myFile
    MeshTopology
    ConstraintOrdering
   )
ADD_EXECUTABLE(${myFile} ${myFile}.cpp)
add_dependencies(${myFile} nuto)
TARGET_LINK_LIBRARIES(${myFile} NuTo ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${ARPACK_LIBRARY})
endforeach(myFile)

