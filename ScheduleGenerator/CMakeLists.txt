cmake_minimum_required(VERSION 3.16)
project(cuCOSMA_Bruteforce)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

add_executable(cuCOSMA_Bruteforce main.cpp Parameters.h Schedule.h Threadblock.h Warptile.h Threadtile.h)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cuCOSMA_Bruteforce PUBLIC OpenMP::OpenMP_CXX)
endif()

