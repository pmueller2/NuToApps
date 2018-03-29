project(ScaledBoundaryFEM)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)

find_package(NuTo)

# look for library files, including default paths
find_library(ARPACK_LIBRARY
  NAMES "arpack"
  PATH_SUFFIXES "lib" "lib32" "lib64"
)

foreach(myFile
    Test
   )
ADD_EXECUTABLE(${myFile} ${myFile}.cpp)
TARGET_LINK_LIBRARIES(${myFile} NuTo ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${ARPACK_LIBRARY})
endforeach(myFile)
