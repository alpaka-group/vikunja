# Copyright 2022 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

add_library(vikunjaBenchSetup INTERFACE)
target_compile_definitions(vikunjaBenchSetup INTERFACE CATCH_CONFIG_ENABLE_BENCHMARKING)
target_include_directories(vikunjaBenchSetup INTERFACE include)
add_library(vikunja::benchSetup ALIAS vikunjaBenchSetup)

target_compile_definitions(vikunjaTestSetup PRIVATE CATCH_CONFIG_ENABLE_BENCHMARKING)

if(VIKUNJA_ENABLE_CUDA_THRUST_BENCHMARKS)
    if(NOT ALPAKA_ACC_GPU_CUDA_ENABLE)
        message(FATAL_ERROR "VIKUNJA_ENABLE_CUDA_THRUST_BENCHMARKS requires the ALPAKA_ACC_GPU_CUDA_ENABLE backend to be enabled.")
    endif()
endif()

add_subdirectory("helper/")
add_subdirectory("transform/")
add_subdirectory("reduce/")
