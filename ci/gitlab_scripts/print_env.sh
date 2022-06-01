#!/bin/bash

# set ouput text color to yellow
echo -e "\e[33m"

echo "export GIT_SUBMODULE_STRATEGY=${GIT_SUBMODULE_STRATEGY}"
env | grep VIKUNJA_CI_

# reset output color
echo -e "\e[0m"
