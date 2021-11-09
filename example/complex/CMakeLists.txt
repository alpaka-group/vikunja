# Copyright 2021 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

project(complexVikunja)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# needs to be a cached variable, otherwise it is overwritten by alpaka
# should be fixed in the future
set(ALPAKA_CXX_STANDARD "17" CACHE STRING "C++ standard version")
set(ALPAKA_CUDA_EXPT_EXTENDED_LAMBDA ON)
set(ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA ON)

add_subdirectory(alpaka)
add_subdirectory(vikunja)

alpaka_add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
  PRIVATE
  vikunja::vikunja
  )
