project(MyPDEs)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)

find_package(NuTo)

foreach(myFile
        MyTestImplicitMethods
        MyTestExplicitMethods
   )
ADD_EXECUTABLE(${myFile} ${myFile}.cpp)
TARGET_LINK_LIBRARIES(${myFile} NuTo ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} )
endforeach(myFile)
