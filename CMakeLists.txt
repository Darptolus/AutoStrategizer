cmake_minimum_required(VERSION 3.27)
project(AutoStrategyzer VERSION 0.1 DESCRIPTION "Communication pattern generation for multi-device collective operations")

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lnuma -fopenmp-targets=nvptx64 --cuda-gpu-arch=sm_70")

if (VERBOSE_MODE)
  message("=> AutoStrategyzer VERBOSE MODE ENABLED")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVERBOSE_MODE=${VERBOSE_MODE}")
  if (FNAME)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFNAME=${FNAME}")  
  endif (FNAME)
endif (VERBOSE_MODE)

include_directories(include)

add_subdirectory( src )
add_subdirectory( tests )
