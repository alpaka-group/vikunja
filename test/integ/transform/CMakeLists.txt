# Copyright 2021 Hauke Mewes, Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

set(_TARGET_NAME "test_transform")


alpaka_add_executable(
  ${_TARGET_NAME}
  src/Transform.cpp
  )

target_include_directories(${_TARGET_NAME} PRIVATE include)

target_link_libraries(${_TARGET_NAME}
  PRIVATE
  vikunja::testSetup
  vikunja::internalvikunja
  )

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_VIKUNJA_TEST_OPTIONS})
