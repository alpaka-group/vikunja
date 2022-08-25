#!/bin/bash

# Copyright 2022 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


: ${VIKUNJA_CI_BOOST_VER?"\e[31mERROR: VIKUNJA_CI_BOOST_VER is not define\e[0m"}
: ${VIKUNJA_CI_CMAKE_VER?"\e[31mERROR: VIKUNJA_CI_CMAKE_VER is not define\e[0m"}
: ${VIKUNJA_CI_CXX?"\e[31mERROR: VIKUNJA_CI_CXX is not define\e[0m"}
: ${VIKUNJA_CI_DEVICE_CXX?"\e[31mERROR: VIKUNJA_CI_DEVICE_CXX is not define\e[0m"}
: ${VIKUNJA_CI_ALPAKA_VER?"\e[31mERROR: VIKUNJA_CI_ALPAKA_VER is not define\e[0m"}

function script_error {
    echo -e "\e[31mERROR: ${1}\e[0m"
    exit 1
}

#####################################
# check BOOST
#####################################

if agc-manager -e boost@${VIKUNJA_CI_BOOST_VER} ; then
    export VIKUNJA_CI_BOOST_ROOT=$(agc-manager -b boost@${VIKUNJA_CI_BOOST_VER})
else
    script_error "No implementation to install boost ${VIKUNJA_CI_BOOST_VER}"
fi

#####################################
# check CMake
#####################################

if agc-manager -e cmake@${VIKUNJA_CI_CMAKE_VER} ; then
    export VIKUNJA_CI_CMAKE_ROOT=$(agc-manager -b cmake@${VIKUNJA_CI_CMAKE_VER})
else
    script_error "No implementation to install cmake ${VIKUNJA_CI_CMAKE_VER}"
fi

#####################################
# check Catch2
#####################################

if agc-manager -e catch2@${VIKUNJA_CI_CATCH_VER} ; then
    export VIKUNJA_CI_CATCH_ROOT=$(agc-manager -b catch2@${VIKUNJA_CI_CATCH_VER})
else
    before_install_catch=$(pwd)
    export VIKUNJA_CI_CATCH_ROOT=/opt/tmpinst/catch2/${VIKUNJA_CI_CATCH_VER}
    cd /tmp
    git clone --branch ${VIKUNJA_CI_CATCH_VER} --depth 1 https://github.com/catchorg/Catch2.git
    cd Catch2
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=${VIKUNJA_CI_CATCH_ROOT}
    cmake --build . -j
    cmake --install .
    cd $before_install_catch
    rm -r /tmp/Catch2
    unset before_install_catch
fi

#####################################
# check host and device compiler
#####################################

function check_compiler {
    compiler_name=$1
    if [[ $compiler_name == g++* ]]; then
        if [ ! -z ${VIKUNJA_CI_GCC_VER+x} ]; then
            if ! agc-manager -e gcc@${VIKUNJA_CI_GCC_VER} ; then
                script_error "No implementation to install gcc ${VIKUNJA_CI_GCC_VER}"
            fi
        else
            script_error "VIKUNJA_CI_GCC_VER is not defined"
        fi
        return
    fi

    if [[ $compiler_name == clang* ]]; then
        if [ ! -z ${VIKUNJA_CI_CLANG_VER+x} ]; then
            if ! agc-manager -e clang@${VIKUNJA_CI_CLANG_VER} ; then
                apt update
                apt install -y llvm-${VIKUNJA_CI_CLANG_VER} clang-${VIKUNJA_CI_CLANG_VER} libomp-${VIKUNJA_CI_CLANG_VER}-dev
            fi
        else
            script_error "VIKUNJA_CI_CLANG_VER is not defined"
        fi
        return
    fi

    if [[ $compiler_name == "nvcc" ]] ; then
        if [ ! -z ${VIKUNJA_CI_NVCC_VER+x} ]; then
            if ! agc-manager -e cuda@${VIKUNJA_CI_NVCC_VER} ; then
                script_error "No implementation to install nvcc@${VIKUNJA_CI_NVCC_VER}"
            fi
        else
            script_error "VIKUNJA_CI_NVCC_VER is not defined"
        fi
        return
    fi

    if [[ $compiler_name == "hipcc" ]] ; then
        if [ ! -z ${VIKUNJA_CI_HIPCC_VER+x} ]; then
            if ! agc-manager -e rocm@${VIKUNJA_CI_HIPCC_VER} ; then
                script_error "No implementation to install nvcc@${VIKUNJA_CI_HIPCC_VER}"
            fi
        else
            script_error "VIKUNJA_CI_HIPCC_VER is not defined"
        fi
        return
    fi

    script_error "unknown compiler -> ${compiler_name}"
}

# check for the host compiler
check_compiler $VIKUNJA_CI_CXX

# check for the device compiler
# skip the check, if host and device compiler are the same
if [ "${VIKUNJA_CI_CXX}" != "${VIKUNJA_CI_DEVICE_CXX}" ]; then
    check_compiler $VIKUNJA_CI_DEVICE_CXX
fi

#####################################
# check installed SDKs
#####################################

if [ ! -z ${VIKUNJA_CI_CUDA_VER+x} ]; then
    if ! agc-manager -e cuda@${VIKUNJA_CI_CUDA_VER} ; then
        script_error "CUDA ${VIKUNJA_CI_CUDA_VER} is not available"
    fi
fi

if [ ! -z ${VIKUNJA_CI_ROCM_VER+x} ]; then
    if ! agc-manager -e rocm@${VIKUNJA_CI_ROCM_VER} ; then
        script_error "ROCm ${VIKUNJA_CI_ROCM_VER} is not available"
    fi
fi

#####################################
# install alpaka
#####################################

base_dir=$(pwd)
git clone --depth 1 --branch ${VIKUNJA_CI_ALPAKA_VER} https://github.com/alpaka-group/alpaka.git
mkdir alpaka/build && cd alpaka/build
${VIKUNJA_CI_CMAKE_ROOT}/bin/cmake .. -DBOOST_ROOT=${VIKUNJA_CI_BOOST_ROOT}
${VIKUNJA_CI_CMAKE_ROOT}/bin/cmake --install .
cd ${base_dir}
rm -rf alpaka
