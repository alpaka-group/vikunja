# Copyright 2021 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

project(cuplaVikReduce)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# needs to be a cached variable, otherwise it is overwritten by alpaka
# should be fixed in the future
set(ALPAKA_CXX_STANDARD "17" CACHE STRING "C++ standard version")
set(ALPAKA_CUDA_EXPT_EXTENDED_LAMBDA ON)
set(ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA ON)

add_subdirectory(alpaka)
add_subdirectory(cupla)
add_subdirectory(vikunja)

###########################################
# test cupla_add_executable only
###########################################

# whole application is build with single cupla_add_excutable
set(target_single_exe ${PROJECT_NAME}SingleExe)

cupla_add_executable(${target_single_exe} src/main.cpp src/cuplaVikReduce.cpp)
target_include_directories(${target_single_exe} PRIVATE include/)
target_link_libraries(${target_single_exe}
  PRIVATE
  cupla::cupla
  vikunja::vikunja
  )

###########################################
# test static linked alpaka_add_executable
###########################################

# application split up in an executable and a static linked library
set(target_static_linked_exe ${PROJECT_NAME}StaticLinkedExe)

cupla_add_executable(${target_static_linked_exe} src/main.cpp)
alpaka_add_library(${target_static_linked_exe}Lib STATIC src/cuplaVikReduce.cpp)
target_include_directories(${target_static_linked_exe}Lib PUBLIC include/)
target_link_libraries(${target_static_linked_exe}Lib
  PRIVATE
  cupla::cupla
  vikunja::vikunja
  )
target_link_libraries(${target_static_linked_exe} PRIVATE ${target_static_linked_exe}Lib)

###########################################
# test dynamic linked alpaka_add_executable
###########################################

# application split up in an executable and a dynamic linked library
set(target_shared_linked_exe ${PROJECT_NAME}DynamicLinkedExe)

cupla_add_executable(${target_shared_linked_exe} src/main.cpp)
alpaka_add_library(${target_shared_linked_exe}Lib SHARED src/cuplaVikReduce.cpp)
target_include_directories(${target_shared_linked_exe}Lib PUBLIC include/)
target_link_libraries(${target_shared_linked_exe}Lib
  PRIVATE
  cupla::cupla
  vikunja::vikunja
  )
target_link_libraries(${target_shared_linked_exe} PRIVATE ${target_shared_linked_exe}Lib)
