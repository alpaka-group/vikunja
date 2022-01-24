# Copyright 2021 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

set(_TARGET_NAME_VIKUNJA_TRANSFORM "bench_vikunja_transform")

alpaka_add_executable(
  ${_TARGET_NAME_VIKUNJA_TRANSFORM}
  bench_vikunja_transform.cpp
  )

target_link_libraries(${_TARGET_NAME_VIKUNJA_TRANSFORM}
  PRIVATE
  vikunja::testSetup
  vikunja::benchSetup
  vikunja::internalvikunja
)

add_test(NAME ${_TARGET_NAME_VIKUNJA_TRANSFORM} COMMAND ${_TARGET_NAME_VIKUNJA_TRANSFORM} ${_VIKUNJA_TEST_OPTIONS})


if(VIKUNJA_ENABLE_CUDA_THRUST_BENCHMARKS)
  set(_TARGET_NAME_THRUST_TRANSFORM "bench_thrust_transform")

  alpaka_add_executable(
    ${_TARGET_NAME_THRUST_TRANSFORM}
    bench_thrust_transform.cpp
  )

  target_link_libraries(${_TARGET_NAME_THRUST_TRANSFORM}
    PRIVATE
    vikunja::testSetup
    vikunja::benchSetup
    vikunja::internalvikunja
  )

  add_test(NAME ${_TARGET_NAME_THRUST_TRANSFORM} COMMAND ${_TARGET_NAME_THRUST_TRANSFORM} ${_VIKUNJA_TEST_OPTIONS})
endif()
