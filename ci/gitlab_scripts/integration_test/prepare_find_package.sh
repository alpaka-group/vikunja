#!/bin/bash

# Copyright 2021 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

set -e
set -o pipefail

echo "CUPLA_TRANSFORM_DIR -> ${CUPLA_TRANSFORM_DIR}"

#####################################
# find CMake
#####################################

if agc-manager -e cmake@${VIKUNJA_CI_CMAKE_VERSION} ; then
    export VIKUNJA_CI_CMAKE_ROOT=$(agc-manager -b cmake@${VIKUNJA_CI_CMAKE_VERSION})
else
    echo "cmake ${VIKUNJA_CI_CMAKE_VERSION} is not available"
    exit 1
fi

###############################################
# find boost
###############################################

if agc-manager -e boost@${VIKUNJA_BOOST_VERSIONS} ; then
    VIKUNJA_BOOST_ROOT=$(agc-manager -b boost@${VIKUNJA_BOOST_VERSIONS})
else
    echo "boost ${VIKUNJA_BOOST_VERSIONS} is not available"
    exit 1
fi

echo "VIKUNJA_BOOST_ROOT -> ${VIKUNJA_BOOST_ROOT}"

###############################################
# Install alpaka
###############################################

cd $CI_PROJECT_DIR
git clone https://github.com/alpaka-group/alpaka.git
cd alpaka
# use this specific alpaka version because of bug fixes for alpaka_add_library
git checkout 261bdf70f359b3d97dfdfb3cc2bd39ec0472c8d1
mkdir build && cd build
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake .. -DBOOST_ROOT=${VIKUNJA_BOOST_ROOT}
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake --build .
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake --install .
cd $CI_PROJECT_DIR
rm -r alpaka

###############################################
# Install cupla
###############################################

cd $CI_PROJECT_DIR
git clone https://github.com/alpaka-group/cupla.git
cd cupla
# use this specific cupla version because of a bug fix in the CMakeLists.txt
git checkout d83fe957009e7b3774f423b3f53887a7af50aabe
mkdir build && cd build
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake .. -DBOOST_ROOT=${VIKUNJA_BOOST_ROOT}
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake --build .
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake --install .
cd $CI_PROJECT_DIR
rm -r cupla

###############################################
# install vikunja
###############################################

cd $CI_PROJECT_DIR
mkdir build && cd build
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake .. -DBOOST_ROOT=${VIKUNJA_BOOST_ROOT}
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake --build .
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake --install .
cd $CI_PROJECT_DIR

###############################################
# prepare project
###############################################

mkdir $CUPLA_TRANSFORM_DIR
cd $CUPLA_TRANSFORM_DIR
cp -r ../example/cuplaReduce/* .
rm CMakeLists.txt
cp ../ci/gitlab_scripts/integration_test/findPackage.cmake CMakeLists.txt

ls
cat CMakeLists.txt

cd $CI_PROJECT_DIR
