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

###############################################
# Install alpaka
###############################################

cd $CI_PROJECT_DIR
git clone https://github.com/alpaka-group/alpaka.git
cd alpaka
# use this specific alpaka version because of bug fixes for alpaka_add_library
git checkout 261bdf70f359b3d97dfdfb3cc2bd39ec0472c8d1
mkdir build && cd build
cmake .. -DBOOST_ROOT=/opt/boost/${VIKUNJA_BOOST_VERSIONS}
cmake --build .
cmake --install .
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
cmake .. -DBOOST_ROOT=/opt/boost/${VIKUNJA_BOOST_VERSIONS}
cmake --build .
cmake --install .
cd $CI_PROJECT_DIR
rm -r cupla

###############################################
# install vikunja
###############################################

cd $CI_PROJECT_DIR
mkdir build && cd build
cmake .. -DBOOST_ROOT=/opt/boost/${VIKUNJA_BOOST_VERSIONS}
cmake --build .
cmake --install .
cd $CI_PROJECT_DIR

###############################################
# prepare project
###############################################

mkdir $CUPLA_TRANSFORM_DIR
cd $CUPLA_TRANSFORM_DIR
cp -r ../example/cuplaReduce/* .
rm CMakeLists.txt
cp ../script/integration_test/findPackage.cmake CMakeLists.txt

ls
cat CMakeLists.txt

cd $CI_PROJECT_DIR
