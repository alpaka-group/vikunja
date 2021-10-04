# Copyright 2019-2021 Hauke Mewes, Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

if(VIKUNJA_SYSTEM_CATCH2)
  find_package(Catch2 REQUIRED)
else()
  message(STATUS "Downloading dependency Catch2.")
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.1
    )
  FetchContent_MakeAvailable(Catch2)
endif()

add_library(vikunjaTestSetup)
target_sources(vikunjaTestSetup
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/CatchMain.cpp
  )
target_include_directories(vikunjaTestSetup
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
target_link_libraries(vikunjaTestSetup
  PUBLIC
  Catch2::Catch2
  )
add_library(vikunja::testSetup ALIAS vikunjaTestSetup)

list(APPEND _VIKUNJA_TEST_OPTIONS "--use-colour yes")
add_subdirectory("unit/")
add_subdirectory("integ/")
