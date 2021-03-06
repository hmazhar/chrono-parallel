#=============================================================================
# CMake configuration file for Chrono_Parallel
#=============================================================================

cmake_minimum_required(VERSION 2.8)

project(ChronoEngine_Parallel)

# Policy CMP0046 warning when a dependency does not exist
# According to a dev from kitware The add_dependencies 
# command is for inter-target ordering dependencies, not
# file-level dependencies.
# This warning appears in Cmake 3.0

if(CMAKE_MAJOR_VERSION EQUAL 3)
    if(COMMAND cmake_policy)
      cmake_policy(SET CMP0046 OLD)
    endif(COMMAND cmake_policy)
endif(CMAKE_MAJOR_VERSION EQUAL 3)

IF(APPLE)
  set(CMAKE_MACOSX_RPATH OFF)
ENDIF()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable CTest
enable_testing()
include(CTest)

# Location of additional CMAKE modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# ------------------------------------------------------------------------------
# User-specified for optional modules
# ------------------------------------------------------------------------------
INCLUDE(CMakeDependentOption)

OPTION(BUILD_UTILS "Build the ChronoEngine_Utils library" ON)
OPTION(BUILD_OPENGL "Build the ChronoEngine_OpenGL library" ON)
OPTION(BUILD_VEHICLE "Enable support for vehicle modeling (requires the Chrono::Vehicle library)" OFF)

# If building the ChronoEngine_Utils library, provide an option to build demo
# programs (default ON). Otherwise, do not build demo programs.
CMAKE_DEPENDENT_OPTION(BUILD_DEMOS "Build demo programs" ON "BUILD_UTILS" OFF)

# ------------------------------------------------------------------------------
# Specify output locations for libraries and executables
# Set the build type depending on the platform
# ------------------------------------------------------------------------------
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

IF (MSVC OR XCODE_VERSION)
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  SET(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "Choose the type of build.")
  mark_as_advanced(FORCE CMAKE_BUILD_TYPE)
  mark_as_advanced(CLEAR CMAKE_CONFIGURATION_TYPES)
ELSE()
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build.")
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
ENDIF()

# ------------------------------------------------------------------------------
# Figure out SSE level support
# ------------------------------------------------------------------------------
OPTION(USE_SSE "Compile with SSE support for floating point math" ON)
OPTION(USE_DOUBLE "Compile with double precision math (no SSE support)" OFF)

IF(USE_DOUBLE)
  OPTION(USE_SSE "Compile with SSE support for floating point math" OFF)
  SET(CHRONO_PARALLEL_USE_DOUBLE "#define CHRONO_PARALLEL_USE_DOUBLE")
  
    IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wa,-q")
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -Wa,-q")
    ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    ENDIF()
  
ELSEIF(USE_SSE)
  FIND_PACKAGE(SSE)

  # Set substitution variables for configuration file.
  IF(SSE_FOUND)
    MESSAGE(STATUS "SSE version: ${SSE_VERSION}")
    SET(CHRONO_PARALLEL_HAS_SSE "#define CHRONO_PARALLEL_HAS_SSE")
    SET(CHRONO_PARALLEL_SSE_LEVEL "#define CHRONO_PARALLEL_SSE_LEVEL \"${SSE_VERSION}\"")
    SET(CHRONO_PARALLEL_SSE_LEVEL "#define CHRONO_PARALLEL_SSE_${SSE_STR}")
  ELSE()
    MESSAGE("No SSE support")
  ENDIF()

  IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
  ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SSE_FLAGS}")
  ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xsse${SSE_VERSION}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -xsse${SSE_VERSION}")
  ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
  ENDIF()
ENDIF()

#Set c++11 flags
IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
# using Clang
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
# using Visual Studio C++
ENDIF()


# ------------------------------------------------------------------------------
# OpenMP
# ------------------------------------------------------------------------------
find_package("OpenMP")

# Add the OpenMP-specific compiler and linker flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

# Determine OpenMP version
IF(OPENMP_FOUND)
  INCLUDE(CheckOpenMPVersion)
  MESSAGE(STATUS "OpenMP version: ${OMP_VERSION}")
  SET(CHRONO_PARALLEL_OMP_FOUND "#define CHRONO_PARALLEL_OMP_FOUND")
  SET(CHRONO_PARALLEL_OMP_VERSION "#define CHRONO_PARALLEL_OMP_VERSION \"${OMP_VERSION}\"")
  IF(OMP_20)
    SET(CHRONO_PARALLEL_OMP_20 "#define CHRONO_PARALLEL_OMP_20")
  ELSE()
    SET(CHRONO_PARALLEL_OMP_20 "")
  ENDIF()
  IF(OMP_30)
    SET(CHRONO_PARALLEL_OMP_30 "#define CHRONO_PARALLEL_OMP_30")
  ELSE()
    SET(CHRONO_PARALLEL_OMP_30 "")
  ENDIF()
  IF(OMP_40)
    SET(CHRONO_PARALLEL_OMP_40 "#define CHRONO_PARALLEL_OMP_40")
  ELSE()
    SET(CHRONO_PARALLEL_OMP_40 "")
  ENDIF()
ENDIF()

# ------------------------------------------------------------------------------
# MPI
# ------------------------------------------------------------------------------
#find_package("MPI" )

# Add the MPI-specific compiler and linker flags
# Also, search for #includes in MPI's paths

#list(APPEND CMAKE_C_COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
#list(APPEND CMAKE_C_LINK_FLAGS ${MPI_C_LINK_FLAGS})
#include_directories(${MPI_C_INCLUDE_PATH})

#list(APPEND CMAKE_CXX_COMPILE_FLAGS ${MPI_CXX_COMPILE_FLAGS})
#list(APPEND CMAKE_CXX_LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
#include_directories(${MPI_CXX_INCLUDE_PATH})

# ------------------------------------------------------------------------------
# Thrust
# ------------------------------------------------------------------------------
find_package(Thrust)

# ------------------------------------------------------------------------------
# BLAZE -- required for some parallel solvers
# ------------------------------------------------------------------------------
IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
SET(BLAZE_DIR ""  CACHE PATH   "Where is Blaze located?")
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
FIND_PATH(BLAZE_DIR NAMES blaze/Blaze.h PATHS "/usr/include" "/usr/local/include")
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
FIND_PATH(BLAZE_DIR NAMES blaze/Blaze.h PATHS "/usr/include" "/usr/local/include")
ENDIF()

# ------------------------------------------------------------------------------
# BOOST -- required for BLAZE
# ------------------------------------------------------------------------------
IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
   SET(BOOST_DIR ""  CACHE PATH   "Where is Boost located?")
ENDIF()

# ------------------------------------------------------------------------------
# Enable the mosek solver
# ------------------------------------------------------------------------------
OPTION(MOSEK_SOLVER "Enable the MOSEK solver (must be installed to use)" OFF)

IF(MOSEK_SOLVER)
  # User specified path to Mosek Header
  SET(MOSEK_INC_PATH           ""    CACHE   PATH    "Where is mosek.h located")
  # User specified path to Mosek Libraries
  SET(MOSEK_LIB_PATH           ""   CACHE    PATH    "Where is Mosek Lib located")
  FIND_LIBRARY(MOSEK_LIB NAMES mosek64 PATHS ${MOSEK_LIB_PATH} REQUIRED)
  INCLUDE_DIRECTORIES(${MOSEK_INC_PATH} )
ENDIF()

# ------------------------------------------------------------------------------
# Find the ChronoEngine library
# ------------------------------------------------------------------------------

FIND_PACKAGE(ChronoEngine)

IF(CHRONOENGINE_LIBRARY_DEBUG)
  MESSAGE(STATUS "ChronoEngine library (debug) found")
ENDIF()
IF(CHRONOENGINE_LIBRARY_RELEASE)
  MESSAGE(STATUS "ChronoEngine library (release) found")
ENDIF()

# Stop now if the path to the Chrono SDK is invalid or if we could not find
# at least one of the debug or release Chrono libraries.
IF(NOT EXISTS "${CH_CHRONO_SDKDIR}/physics/ChSystem.h")
  MESSAGE(FATAL_ERROR "Cannot find the Chrono SDK. Make sure CH_CHRONO_SDKDIR is set correctly.")
  RETURN()
ENDIF()
IF(NOT CHRONOENGINE_LIBRARY_DEBUG AND NOT CHRONOENGINE_LIBRARY_RELEASE)
  MESSAGE(FATAL_ERROR "Cannot find the ChronoEngine library. Check CH_LIBDIR_DEBUG and CH_LIBDIR_RELEASE.")
  RETURN()
ENDIF()

# Add paths to Chrono headers
INCLUDE_DIRECTORIES(${CHRONOENGINE_INCLUDES})

# ------------------------------------------------------------------------------
# Find the ChronoVehicle library
# ------------------------------------------------------------------------------
IF(BUILD_VEHICLE)

  FIND_PACKAGE(ChronoVehicle)

  IF(CHRONOVEHICLE_LIBRARY_DEBUG)
    MESSAGE(STATUS "ChronoVehicle library (debug) found")
  ENDIF()
  IF(CHRONOVEHICLE_LIBRARY_RELEASE)
    MESSAGE(STATUS "ChronoVehicle library (release) found")
  ENDIF()

  SET(ENABLE_VEHICLE ON)

  # Disable ChronoVehicle support if we could not find the ChronoVehicle SDK or
  # if we could not find one of the debug or release ChronoVehicle libraries.
  IF(NOT EXISTS "${CH_VEHICLE_SDKDIR}/subsys/ChVehicle.h")
    MESSAGE("Cannot find the ChronoVehicle SDK. Make sure CH_VEHICLE_SDKDIR is set correctly.")
    SET(ENABLE_VEHICLE OFF)
  ENDIF()
  IF(NOT CHRONOVEHICLE_LIBRARY_DEBUG AND NOT CHRONOVEHICLE_LIBRARY_RELEASE)
    MESSAGE("Cannot find the ChronoVehicle library. Check CH_VEHICLE_LIBDIR_DEBUG and CH_VEHICLE_LIBDIR_RELEASE.")
    SET(ENABLE_VEHICLE OFF)
  ENDIF()

ELSE()

  SET(ENABLE_VEHICLE OFF)

ENDIF()

IF(ENABLE_VEHICLE)

  # Add path to ChronoVehicle headers.
  INCLUDE_DIRECTORIES(${CHRONOVEHICLE_INCLUDES})

  # Prepare substitution variables for configuration header.
  SET(CHRONO_PARALLEL_HAS_VEHICLE "#define CHRONO_PARALLEL_HAS_VEHICLE")
  SET(CHRONO_VEHICLE_DATA_DIR "${CH_VEHICLE_SDKDIR}/data/")

ELSE()

  # Dummy substitution variables.
  SET(CHRONO_PARALLEL_HAS_VEHICLE "")
  SET(CHRONO_VEHICLE_DATA_DIR "")
 
ENDIF()

# ------------------------------------------------------------------------------
# Dependencies for ChronoEngine_OpenGL
# ------------------------------------------------------------------------------
IF(BUILD_OPENGL)

  find_package(OpenGL)
  find_package(GLM)
  find_package(GLEW)
  find_package(GLFW)

  # On windows, ask for the GLEW and GLFW DLLs so that we can copy. This is
  # optional.  If not specified, it is the user's responsibility to make them
  # available at runtime.
  IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    SET(GLEW_DLL  "" CACHE FILEPATH "The GLEW DLL")
    SET(GLFW_DLL  "" CACHE FILEPATH "The GLFW DLL")
  ENDIF()

  # If all dependencies are available, enable building OpenGL support.
  IF(OPENGL_FOUND AND GLM_FOUND AND GLEW_FOUND AND GLFW_FOUND)
    SET(CHRONO_PARALLEL_HAS_OPENGL "#define CHRONO_PARALLEL_HAS_OPENGL")
    SET(ENABLE_OPENGL ON)
  ELSE()
    # Unable to find everything, turn off OpenGL
    MESSAGE("Cannot build ChronoEngine_OpenGL (missing dependencies).")
    SET(CHRONO_PARALLEL_HAS_OPENGL "")
    SET(ENABLE_OPENGL OFF)
  ENDIF()

ELSE()

  SET(CHRONO_PARALLEL_HAS_OPENGL "")
  SET(ENABLE_OPENGL OFF)

ENDIF()


# ------------------------------------------------------------------------------
# Automatically copy DLLs 
# ------------------------------------------------------------------------------
IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  # Get the path to the Chrono bin directory (at this point, we know that at
  # least one of CH_LIBDIR_DEBUG or CH_LIBDIR_RELEASE is properly specified)
  IF(CH_LIBDIR_DEBUG)
      SET(CH_BINDIR "${CH_LIBDIR_DEBUG}/../../bin")
  ENDIF()
  IF(CH_LIBDIR_RELEASE)
      SET(CH_BINDIR "${CH_LIBDIR_RELEASE}/../../bin")
  ENDIF()

  # Create custom target for copying DLLs; add it to the default build target
  ADD_CUSTOM_TARGET(COPY_DLLS ALL)

  # Create custom commands, invoked post-build to copy DLLs to the appropriate
  # directory (depending on the configuration selected at build time in VS)
  ADD_CUSTOM_COMMAND(
    TARGET COPY_DLLS POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CH_BINDIR}/$<CONFIGURATION>/ChronoEngine.dll"
        "${CMAKE_BINARY_DIR}/bin/$<CONFIGURATION>"
  )

  # If ChronoVehicle support is enabled, copy the ChronoVehicle DLL
  IF(ENABLE_VEHICLE)
    IF(CH_VEHICLE_LIBDIR_DEBUG)
        SET(CH_VEHICLE_BINDIR "${CH_VEHICLE_LIBDIR_DEBUG}/../../bin")
    ENDIF()
    IF(CH_VEHICLE_LIBDIR_RELEASE)
        SET(CH_VEHICLE_BINDIR "${CH_VEHICLE_LIBDIR_RELEASE}/../../bin")
    ENDIF()
    ADD_CUSTOM_COMMAND(
        TARGET COPY_DLLS POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CH_VEHICLE_BINDIR}/$<CONFIGURATION>/ChronoVehicle.dll"
            "${CMAKE_BINARY_DIR}/bin/$<CONFIGURATION>"
    )
  ENDIF()

  # If building OpenGL support, also copy the DLLs for GLEW and GLFW (if specified)
  IF(ENABLE_OPENGL)
    IF(EXISTS ${GLEW_DLL})
      ADD_CUSTOM_COMMAND(
        TARGET COPY_DLLS POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLEW_DLL}"
            "${CMAKE_BINARY_DIR}/bin/$<CONFIGURATION>"
      )
    ENDIF()
    IF(EXISTS ${GLFW_DLL})
      ADD_CUSTOM_COMMAND(
        TARGET COPY_DLLS POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLFW_DLL}"
            "${CMAKE_BINARY_DIR}/bin/$<CONFIGURATION>"
      )
    ENDIF()
  ENDIF()

ENDIF()


# ----------------------------------------------------------------------------
# Generate configuration header file.
# ----------------------------------------------------------------------------
# Generate the configuration header file using substitution variables.
# Place the header file in the library output directory and make sure it can
# be found at compile time.

CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/cmake/ChConfigParallel.in
  ${PROJECT_BINARY_DIR}/chrono_parallel/ChConfigParallel.h
  )

INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR} ${BLAZE_DIR} ${BOOST_DIR} ${THRUST_INCLUDE_DIR})

# ------------------------------------------------------------------------------
# Add paths to the top of the source directory
# ------------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

# ------------------------------------------------------------------------------
# Add subdirectories, depending on user-specified options
# ------------------------------------------------------------------------------
ADD_SUBDIRECTORY(chrono_parallel)

IF(BUILD_UTILS)
  ADD_SUBDIRECTORY(chrono_utils)
ENDIF()

IF(ENABLE_OPENGL)
  ADD_SUBDIRECTORY(chrono_opengl)
ENDIF()

IF(BUILD_DEMOS)
  ADD_SUBDIRECTORY(tests)
ENDIF()

IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(unit_testing)
ENDIF()
