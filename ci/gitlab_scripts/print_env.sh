#!/bin/bash

# Copyright 2022 Simeon Ehrig
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# set ouput text color to yellow
echo -e "\e[33m"

echo "export GIT_SUBMODULE_STRATEGY=${GIT_SUBMODULE_STRATEGY}"
env | grep VIKUNJA_CI_

# reset output color
echo -e "\e[0m"
