project(Waves)
cmake_minimum_required(VERSION 3.0)
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
        plateH2_angle45_L0.4.msh
        plateH2_angle45_L0.4small.msh
        plateH1_angle45_L0.4.msh
        plateH0.3_angle45_L0.4small.msh
        plateH0.5_angle45_L0.4.msh
        plateH0.75_angle45_L0.4.msh
        plateH2_angle45_L0.4smallLinear.msh
        cube20x20x20.msh
    )
    execute_process(COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink"
        "${CMAKE_CURRENT_SOURCE_DIR}/${inputFile}"
        "${CMAKE_CURRENT_BINARY_DIR}/${inputFile}")

endforeach()

include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/lib")

foreach(myFile
    ElasticityCrackOpening2D
    ElasticityCrackOpening3D
    Wave1D
    mytest
    GenerateReferenceMatrices
   )
ADD_EXECUTABLE(${myFile} ${myFile}.cpp)
add_dependencies(${myFile} nuto)
TARGET_LINK_LIBRARIES(${myFile} NuTo ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${ARPACK_LIBRARY})
endforeach(myFile)

