project(Waves)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")

FIND_PACKAGE( OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


find_path(EIGEN_INCLUDE_DIR NAMES Eigen/Core PATH_SUFFIXES eigen3)
include_directories(${EIGEN_INCLUDE_DIR})

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
        plateH0.1_angle45_L0.4small.msh
    )
    execute_process(COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink"
        "${CMAKE_CURRENT_SOURCE_DIR}/Meshes/${inputFile}"
        "${CMAKE_CURRENT_BINARY_DIR}/${inputFile}")

endforeach()

foreach(myFile
    ElasticityCrackOpening2D
    ElasticityCrackOpening3D
    PointForce3D
    mytest
    GenerateReferenceMatrices
    AssemblyBenchmark
   )
ADD_EXECUTABLE(${myFile} ${myFile}.cpp)
add_dependencies(${myFile} nuto)
target_include_directories(${myFile} BEFORE PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include")
TARGET_LINK_LIBRARIES(${myFile} NuTo ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${ARPACK_LIBRARY})
endforeach(myFile)


ADD_EXECUTABLE(Wave1D Wave1D.cpp ScalarWaveEquation.cpp)
add_dependencies(Wave1D nuto)
TARGET_LINK_LIBRARIES(Wave1D NuTo ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${ARPACK_LIBRARY})

