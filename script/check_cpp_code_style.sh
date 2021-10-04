#!/bin/bash

# Copyright 2021 René Widera
#
# This file is part of vikunja.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

set -e
set -o pipefail

cd $CI_PROJECT_DIR

# check code style with clang format
find example include test  -iname "*.def" \
  -o -iname "*.h" -o -iname "*.cpp" -o -iname "*.hpp" \
  | xargs clang-format-11 --dry-run --Werror
