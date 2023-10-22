# Copyright 2019-2021 Hauke Mewes, Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

set(_TARGET_NAME_MEMOBJ "test_memobj")

alpaka_add_executable(
    ${_TARGET_NAME_MEMOBJ}
    src/memObj.cpp
)

target_link_libraries(${_TARGET_NAME_MEMOBJ}
    PRIVATE
    vikunja::internalvikunja
    vikunja::testSetup
)

add_test(NAME ${_TARGET_NAME_MEMOBJ} COMMAND ${_TARGET_NAME_MEMOBJ} ${_VIKUNJA_TEST_OPTIONS})
