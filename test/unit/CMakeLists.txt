# Copyright 2021 Hauke Mewes, Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

add_subdirectory("iterator/")
if(VIKUNJA_ENABLE_CXX_TEST)
  add_subdirectory("cxx/")
endif()
