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

cd $CUPLA_TRANSFORM_DIR
mkdir build && cd build
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake .. -DBOOST_ROOT=${VIKUNJA_BOOST_ROOT} -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON
$VIKUNJA_CI_CMAKE_ROOT/bin/cmake --build .
./cuplaVikReduceSingleExe
./cuplaVikReduceStaticLinkedExe
./cuplaVikReduceDynamicLinkedExe
