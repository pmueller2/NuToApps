project(Waves)
cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")

find_path(EIGEN_INCLUDE_DIR NAMES Eigen/Core PATH_SUFFIXES eigen3)
include_directories(${EIGEN_INCLUDE_DIR})
include_directories("include")

find_package(Boost COMPONENTS filesystem REQUIRED)

include(ExternalProject)

ExternalProject_Add(nuto
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}"
               "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../NuTo"
    BUILD_ALWAYS 1
    )

foreach(inputFile
        Crack2D_1.msh
        Crack2D_2.msh
        Crack3D_angle00_h1.msh
        Crack3D_angle00_h1_2ndOrder.msh
        Crack3D_angle00_h2.msh
        Crack3D_angle45_h1.msh
        Crack3D_angle45_h1_2ndOrder.msh
        Crack3D_angle45_h2.msh
        Crack3D_angle45_h2_2ndOrder.msh
        Crack3D_angle90_h1.msh
        Crack3D_angle90_h1_2ndOrder.msh
        Crack3D_angle90_h2.msh
    )
    execute_process(COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink"
        "${CMAKE_CURRENT_SOURCE_DIR}/${inputFile}"
        "${CMAKE_CURRENT_BINARY_DIR}/${inputFile}")

endforeach()

link_directories("${CMAKE_CURRENT_BINARY_DIR}/lib")

foreach(myFile
    ElasticityCrackOpening2D
    ElasticityCrackOpening3D
    ElasticityCrackOpening3D_00
    ElasticityCrackOpening3D_45
    ElasticityCrackOpening3D_90
    Wave1D
    mytest
   )
ADD_EXECUTABLE(${myFile} ${myFile}.cpp)
add_dependencies(${myFile} nuto)
TARGET_LINK_LIBRARIES(${myFile} NuTo ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${ARPACK_LIBRARY})
endforeach(myFile)

