# Copyright 2022 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Create an executable file for the test, link it to the required test targets and add it to ctest.
#
#    vikunja_add_default_test(TARGET <name>
#                             SOURCE <list of source files>
#                             [INCLUDE <list of include paths>]
#                            )
#
# * TARGET: name of the executable, will be prefixed with `test_`
# * SOURCE: path of the source file(s)
# * INCLUDE: optional path(s) to include folder(s)
macro(vikunja_add_default_test)
  set(_MACRO_PREFIX "vikTest")
  set(_SINGLE_ARG TARGET)
  set(_MULTI_ARG SOURCE INCLUDE)

  cmake_parse_arguments(
    "${_MACRO_PREFIX}"
    "" # option
    "${_SINGLE_ARG}"
    "${_MULTI_ARG}"
    "${ARGN}"
    )

  if(NOT DEFINED ${_MACRO_PREFIX}_TARGET)
    message(FATAL_ERROR "vikunja_add_default_test: no target name defined")
  endif()

  if(NOT DEFINED ${_MACRO_PREFIX}_SOURCE)
    message(FATAL_ERROR "vikunja_add_default_test: no source files defined")
  endif()

  set(_TARGET_NAME "test_${${_MACRO_PREFIX}_TARGET}")

  alpaka_add_executable(
    ${_TARGET_NAME}
    ${${_MACRO_PREFIX}_SOURCE}
  )

  if(DEFINED ${_MACRO_PREFIX}_INCLUDE)
    message(VERBOSE "vikunja_add_default_test: add ${${_MACRO_PREFIX}_INCLUDE} include paths to ${_TARGET_NAME}")
    target_include_directories(${_TARGET_NAME} PRIVATE ${${_MACRO_PREFIX}_INCLUDE})
  endif()

  target_link_libraries(${_TARGET_NAME}
    PRIVATE
    vikunja::internalvikunja
    vikunja::testSetup
  )

  add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_VIKUNJA_TEST_OPTIONS})

endmacro()
