#!/bin/bash

# Copyright 2022 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

: ${VIKUNJA_CI_ALPAKA_BACKENDS?"VIKUNJA_CI_ALPAKA_BACKENDS must be specified"}
: ${VIKUNJA_CI_CXX?"VIKUNJA_CI_CXX must be specified"}
: ${VIKUNJA_CI_CXX_STANDARD?"VIKUNJA_CI_CXX_STANDARD must be specified"}
: ${VIKUNJA_CI_DEVICE_CXX?"VIKUNJA_CI_DEVICE_CXX must be specified"}
: ${VIKUNJA_CI_EXTRA_ARGS?"VIKUNJA_CI_EXTRA_ARGS must be specified"}
: ${VIKUNJA_CI_BOOST_ROOT?"VIKUNJA_CI_EXTRA_ARGS must be specified"}
: ${VIKUNJA_CI_CMAKE_ROOT?"VIKUNJA_CI_EXTRA_ARGS must be specified"}
: ${VIKUNJA_CI_CONST_ARGS?"VIKUNJA_CI_CONST_ARGS must be specified"}
: ${VIKUNJA_CI_ALPAKA_VER?"VIKUNJA_CI_ALPAKA_VER must be specified"}


cmake_args=""

cmake_args="${cmake_args} -DCMAKE_CXX_COMPILER=${VIKUNJA_CI_CXX}"
cmake_args="${cmake_args} -Dalpaka_CXX_STANDARD=${VIKUNJA_CI_CXX_STANDARD}"
cmake_args="${cmake_args} -DBOOST_ROOT=${VIKUNJA_CI_BOOST_ROOT}"
cmake_args="${cmake_args} -DVIKUNJA_ENABLE_CXX_TEST=${VIKUNJA_CI_CXX_TEST}"
cmake_args="${cmake_args} ${VIKUNJA_CI_ALPAKA_BACKENDS}"

# if the nvcc is the device compiler, set the correct host and device compiler
if [ $VIKUNJA_CI_DEVICE_CXX == "nvcc" ]; then
    cmake_args="${cmake_args} -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_HOST_COMPILER=${VIKUNJA_CI_CXX}"
fi

# if the clang is the CUDA device compiler, set the correct device compiler
if [ ! -z ${VIKUNJA_CI_CUDA_VER+x} ]; then
    if [[ $VIKUNJA_CI_DEVICE_CXX == clang* ]]; then
        cmake_args="${cmake_args} -DCMAKE_CUDA_COMPILER=clang++"
    fi
fi

cmake_args="${cmake_args} ${VIKUNJA_CI_CONST_ARGS}"
cmake_args="${cmake_args} ${VIKUNJA_CI_EXTRA_ARGS}"

echo -e "\033[0;32m///////////////////////////////////////////////////"
echo "number of processor threads -> $(nproc)"
echo "ALPAKA_VERSION -> ${VIKUNJA_CI_ALPAKA_VER}"
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake --version | head -n 1
echo "CMAKE_ARGS -> ${cmake_args}"
echo -e "/////////////////////////////////////////////////// \033[0m \n\n"


# use one build directory for all build configurations
mkdir -p build
cd build

$VIKUNJA_CI_CMAKE_ROOT/bin/cmake .. $cmake_args
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake --build . -j
$VIKUNJA_CI_CMAKE_ROOT/bin/ctest --output-on-failure
# if ALPAKA_CXX_STANDARD is set manually, run this without ctest
# to eliminate errors in CMake
if [ "${VIKUNJA_CI_CXX_TEST}" == "ON" ]; then
    echo -e "test/unit/cxx/test_cxx --cxx ${VIKUNJA_CI_CXX_STANDARD}"
    test/unit/cxx/test_cxx --cxx ${VIKUNJA_CI_CXX_STANDARD}
fi
