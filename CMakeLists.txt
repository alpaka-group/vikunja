# Copyright 2021 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)
project(vikunja
  VERSION      0.1.0
  DESCRIPTION  "Primitives for Alpaka"
  HOMEPAGE_URL "https://github.com/alpaka-group/vikunja"
  LANGUAGES    CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED 14)

include(CTest)

option(vikunja_BUILD_EXAMPLES "Build the examples" OFF)
option(vikunja_BUILD_DOXYGEN "Build the doxygen documentation" OFF)
option(vikunja_ENABLE_EXTRA_WARING "Enable extra warnings" ON)
option(VIKUNJA_REDUCE_COMPARING_BENCHMARKS_ENABLE "Should the reduce integration test run some comparison benchmarks?" OFF)
option(VIKUNJA_TRANSFORM_COMPARING_BENCHMARKS_ENABLE "Should the transform integration test run some comparison benchmarks?" OFF)

# activate support for host/device lambdas in cuda
# needs to be set before alpaka is included
if(ALPAKA_ACC_GPU_CUDA_ENABLE)
  # DEPRECATED: used until alpaka 0.6.x
  set(ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA ON)
  # used with alpaka 0.7.0 and below
  set(ALPAKA_CUDA_EXPT_EXTENDED_LAMBDA ON)
endif()

# the sequential accelerator is required for the tests and examples
if(NOT DEFINED ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE AND (vikunja_BUILD_EXAMPLES OR BUILD_TESTING))
  option(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE "enable alpaka serial accelerator" ON)
endif()

set(_VIKUNJA_MIN_ALPAKA_VERSION 0.6.0)
set(_VIKUNJA_UNSUPPORTED_ALPAKA_VERSION 0.8.0)

if(NOT TARGET alpaka::alpaka)
  find_package(alpaka ${_VIKUNJA_MIN_ALPAKA_VERSION} REQUIRED HINTS $ENV{ALPAKA_ROOT})
  if(alpaka_VERSION VERSION_GREATER_EQUAL _VIKUNJA_UNSUPPORTED_ALPAKA_VERSION)
    message(WARNING "Unsupported alpaka version ${alpaka_VERSION}. "
        "Supported versions [${_VIKUNJA_MIN_ALPAKA_VERSION},${_VIKUNJA_UNSUPPORTED_ALPAKA_VERSION}).")
  endif()
endif()

if(ALPAKA_ACC_GPU_CUDA_ENABLE)
    # activate support for host/device lambdas in cuda, currently (as of CUDA 10) still an experimental feature,
    # but it seems to work.
    # PARENT_SCOPE is necessary, if the project is imported via add_subdirectory()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda" PARENT_SCOPE)
endif()


add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

#The target vikunja::internalvikunja is for internal use only, such as the tests or examples.
# For external projects that use vikunja, the target vikunja::vikunja should be used.

# The internal target links alpaka to make it easier to use vikunja, but the target cannot be
# exported because all alpaka properties would be saved at install time and changing
# the properties in external projects would not be possible.
# For example, only the accelerators that are enabled when vikunja is installed are available
# for external projects. Changing them in the external project is not possible.
if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  add_library(internal${PROJECT_NAME} INTERFACE)
  target_link_libraries(internal${PROJECT_NAME}
    INTERFACE
    vikunja::vikunja
    alpaka::alpaka)
  add_library(${PROJECT_NAME}::internal${PROJECT_NAME} ALIAS internal${PROJECT_NAME})
else()
  # if the project is imported via add_subdirectory() in another project
  # the linkage of alpaka have to be done here
  target_link_libraries(${PROJECT_NAME}
    INTERFACE
    alpaka::alpaka)
endif()

if(vikunja_ENABLE_EXTRA_WARING)
  if(MSVC)
    target_compile_options(${PROJECT_NAME} INTERFACE "/W4")
  else()
    target_compile_options(${PROJECT_NAME} INTERFACE "-Wall;-Wextra;-pedantic")
  endif()
endif()


if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  #########################################
  ######### Install Configuration #########
  #########################################

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION 0.1
    COMPATIBILITY AnyNewerVersion
    )

  include(GNUInstallDirs)
  # adding alpaka to the install could be a workaround
  # without, the target alpaka::alpaka is not available
  # does not install the alapaka library itself
  install(TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Targets"
    )

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )

  install(EXPORT "${PROJECT_NAME}Targets"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    NAMESPACE ${PROJECT_NAME}::)
  install(FILES
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

  #########################################
  ############# Documentation #############
  #########################################

  # Use doxygen for api doc when available.
  if(vikunja_BUILD_DOXYGEN)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(doc include/)
  endif()

  #########################################
  ################ Testing ################
  #########################################

  if(BUILD_TESTING)
    enable_testing()
    add_subdirectory("test/")
  endif()

  #########################################
  ################ Examples ###############
  #########################################

  if(vikunja_BUILD_EXAMPLES)
    add_subdirectory("example/")
  endif()
endif()
