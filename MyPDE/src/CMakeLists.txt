project(MyPDEs)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)

find_package(NuTo)

# look for library files, including default paths
find_library(ARPACK_LIBRARY
  NAMES "arpack"
  PATH_SUFFIXES "lib" "lib32" "lib64"
)

foreach(inputFile
        square0.01.msh
        disk02.msh
        cube5.msh
        cube1.msh
        diskRegular_middle.msh
        plateWithInternalCrackHexed.msh
        plateWithInternalCrackHexedOrder2.msh
        plateWithInternalCrack0.5Hexed2ndOrder.msh
        rectangle100x100.msh
        rectangleWithInternalCrack01.msh
        circle.msh
        annulus.msh
    )
    execute_process(COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink"
        "${CMAKE_CURRENT_SOURCE_DIR}/${inputFile}"
        "${CMAKE_CURRENT_BINARY_DIR}/${inputFile}")

endforeach()

foreach(myFile
    Coordinates
    TestNormal
    PiezoelectricDisk
    LocalCoordinatesFromGlobal
    LaplaceRectangle
    LaplacePolarCoordinates
    LaplaceLine
    LaplaceCircle
    InterpolationHierarchical
    LaplaceAnnulus
    PoissonLine
    WaveLine
    Interpolation1DTests
    Wave1DTests
    SimplexShapes
    ElasticityRectangleDynamic
    ElasticityDiskDynamic
    ElasticityLShapedDomain
    InterpolationTrigonometricTest
    ElasticityPlateTiltedCrack
    ElasticityRectangleTiltedCrack
    HelmholtzRectangle
    PoissonRectangle
    Electrostatics
   )
ADD_EXECUTABLE(${myFile} ${myFile}.cpp)
TARGET_LINK_LIBRARIES(${myFile} NuTo ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${ARPACK_LIBRARY})
endforeach(myFile)
