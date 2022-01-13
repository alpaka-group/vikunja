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

set(ALPAKA_CXX_STANDARD "14" CACHE STRING "C++ standard version")

include(CMakeDependentOption)
include(FetchContent)

option(VIKUNJA_BUILD_EXAMPLES "Build the examples" OFF)
option(VIKUNJA_ENABLE_EXTRA_WARNINGS "Enable extra warnings" OFF)
option(BUILD_TESTING "Build the testing tree." OFF)
cmake_dependent_option(VIKUNJA_SYSTEM_CATCH2 "Use your local installation of Catch2" ON BUILD_TESTING OFF)
cmake_dependent_option(VIKUNJA_ENABLE_CXX_TEST "Builds test that checks if the C++ standard is set correctly" OFF BUILD_TESTING OFF)

# activate support for host/device lambdas in cuda
# needs to be set before alpaka is included
if(ALPAKA_ACC_GPU_CUDA_ENABLE)
  # DEPRECATED: used until alpaka 0.6.x
  set(ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA ON)
  # used with alpaka 0.7.0 and later
  set(ALPAKA_CUDA_EXPT_EXTENDED_LAMBDA ON)
endif()

# the sequential accelerator is required for the tests and examples
if(NOT DEFINED ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE AND (VIKUNJA_BUILD_EXAMPLES OR BUILD_TESTING))
  option(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE "enable alpaka serial accelerator" ON)
endif()

set(_VIKUNJA_MIN_ALPAKA_VERSION 0.6.0)
set(_VIKUNJA_UNSUPPORTED_ALPAKA_VERSION 0.9.0)

if(NOT TARGET alpaka::alpaka)
  find_package(alpaka ${_VIKUNJA_MIN_ALPAKA_VERSION} REQUIRED HINTS $ENV{ALPAKA_ROOT})
  if(alpaka_VERSION VERSION_GREATER_EQUAL _VIKUNJA_UNSUPPORTED_ALPAKA_VERSION)
    message(WARNING "Unsupported alpaka version ${alpaka_VERSION}. "
        "Supported versions [${_VIKUNJA_MIN_ALPAKA_VERSION},${_VIKUNJA_UNSUPPORTED_ALPAKA_VERSION}).")
  endif()
endif()

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
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

if(VIKUNJA_ENABLE_EXTRA_WARNINGS)
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
  ################ Testing ################
  #########################################

  if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory("test/")
  endif()

  #########################################
  ################ Examples ###############
  #########################################

  if(VIKUNJA_BUILD_EXAMPLES)
    add_subdirectory("example/")
  endif()
endif()
