# Copyright 2019-2021 Hauke Mewes, Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

set(_TARGET_NAME_OPERATOR "test_concept_operator")

alpaka_add_executable(
  ${_TARGET_NAME_OPERATOR}
  src/operator.cpp
)

target_link_libraries(${_TARGET_NAME_OPERATOR}
  PRIVATE
  vikunja::internalvikunja
  vikunja::testSetup
)

add_test(NAME ${_TARGET_NAME_OPERATOR} COMMAND ${_TARGET_NAME_OPERATOR} ${_VIKUNJA_TEST_OPTIONS})

set(_TARGET_NAME_ACC "test_concept_acc")

alpaka_add_executable(
  ${_TARGET_NAME_ACC}
  src/acc.cpp
)

target_link_libraries(${_TARGET_NAME_ACC}
  PRIVATE
  vikunja::internalvikunja
  vikunja::testSetup
)

add_test(NAME ${_TARGET_NAME_ACC} COMMAND ${_TARGET_NAME_ACC} ${_VIKUNJA_TEST_OPTIONS})

set(_TARGET_NAME_MEM_VISIBILITY "test_concept_memVisibility")

alpaka_add_executable(
  ${_TARGET_NAME_MEM_VISIBILITY}
  src/memVisibility.cpp
)

target_link_libraries(${_TARGET_NAME_MEM_VISIBILITY}
  PRIVATE
  vikunja::internalvikunja
  vikunja::testSetup
)

add_test(NAME ${_TARGET_NAME_MEM_VISIBILITY} COMMAND ${_TARGET_NAME_MEM_VISIBILITY} ${_VIKUNJA_TEST_OPTIONS})
