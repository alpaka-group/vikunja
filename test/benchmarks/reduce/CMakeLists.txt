# Copyright 2022 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

set(_TARGET_NAME_VIKUNJA_REDUCE "bench_vikunja_reduce")

alpaka_add_executable(
  ${_TARGET_NAME_VIKUNJA_REDUCE}
  bench_vikunja_reduce.cpp
  )

target_link_libraries(${_TARGET_NAME_VIKUNJA_REDUCE}
  PRIVATE
  vikunja::testSetup
  vikunja::benchSetup
  vikunja::internalvikunja
)

add_test(NAME ${_TARGET_NAME_VIKUNJA_REDUCE} COMMAND ${_TARGET_NAME_VIKUNJA_REDUCE} ${_VIKUNJA_TEST_OPTIONS})
# avoid running the benchmarks in parallel
set_tests_properties(${_TARGET_NAME_VIKUNJA_REDUCE} PROPERTIES RUN_SERIAL TRUE)


set(_TARGET_NAME_VIKUNJA_REDUCE "bench_vikunja_constiter_reduce")

alpaka_add_executable(
  ${_TARGET_NAME_VIKUNJA_REDUCE}
  bench_vikunja_constiter_reduce.cpp
  )

target_link_libraries(${_TARGET_NAME_VIKUNJA_REDUCE}
  PRIVATE
  vikunja::testSetup
  vikunja::benchSetup
  vikunja::internalvikunja
)

add_test(NAME ${_TARGET_NAME_VIKUNJA_REDUCE} COMMAND ${_TARGET_NAME_VIKUNJA_REDUCE} ${_VIKUNJA_TEST_OPTIONS})
# avoid running the benchmarks in parallel
set_tests_properties(${_TARGET_NAME_VIKUNJA_REDUCE} PROPERTIES RUN_SERIAL TRUE)


if(VIKUNJA_ENABLE_CUDA_THRUST_BENCHMARKS)
  set(_TARGET_NAME_THRUST_REDUCE "bench_thrust_reduce")

  alpaka_add_executable(
    ${_TARGET_NAME_THRUST_REDUCE}
    bench_thrust_reduce.cpp
  )

  target_link_libraries(${_TARGET_NAME_THRUST_REDUCE}
    PRIVATE
    vikunja::testSetup
    vikunja::benchSetup
    vikunja::internalvikunja
  )

  add_test(NAME ${_TARGET_NAME_THRUST_REDUCE} COMMAND ${_TARGET_NAME_THRUST_REDUCE} ${_VIKUNJA_TEST_OPTIONS})
  set_tests_properties(${_TARGET_NAME_THRUST_REDUCE} PROPERTIES RUN_SERIAL TRUE)
endif()
