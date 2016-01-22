include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cmake_minimum_required(VERSION 2.6)
project(control)

option(DEVEL_MODE "build with compiler more errors" OFF)
option(HAVE_MJC2 "have mjc2" OFF)

option(USE_GCC "build with gcc" OFF)
if (GCC)
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
  endif()


set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")

# http://cmake.3232098.n2.nabble.com/Default-value-for-CMAKE-BUILD-TYPE-td7550756.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib 
    CACHE PATH "Output directory for static libraries.")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CACHE PATH "Output directory for shared libraries.")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CACHE PATH "Output directory for executables and DLL's.")


#set(BUILD_SHARED_LIBS true)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


# find_package(Eigen REQUIRED)
# add_definitions(-DEIGEN_DEFAULT_TO_ROW_MAJOR)
# include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

include(${CMAKE_SOURCE_DIR}/cmake/UseCython.cmake)
find_package(Boost COMPONENTS python REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Numpy)

include(${CMAKE_SOURCE_DIR}/cmake/boost-python.cmake)


if (APPLE)
  include_directories(SYSTEM "/opt/X11/include")
  link_directories("/opt/X11/lib")
endif(APPLE)


add_subdirectory(Boost.NumPy)
add_subdirectory(../private/mujoco_sources/mjc "${CMAKE_CURRENT_BINARY_DIR}/mjc")
add_subdirectory(../private/mujoco_sources/mjc2 "${CMAKE_CURRENT_BINARY_DIR}/mjc2")
#add_subdirectory(../private/mujoco/src/igor_mjc "${CMAKE_CURRENT_BINARY_DIR}/igor_mjc")
add_subdirectory(../private/mujoco/src/mjcpy "${CMAKE_CURRENT_BINARY_DIR}/mjcpy")
add_subdirectory(../private/mujoco/src/mjcpy2 "${CMAKE_CURRENT_BINARY_DIR}/mjcpy2")
