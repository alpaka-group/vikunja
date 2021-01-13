# Primitives for Alpaka
This contains a `reduce` and a `transform` primitive for Alpaka, both header-only.
Directories:
* `include/vikunja`: The actual code
    - `mem/iterator`: Contains a base iterator class and a policy based iterator that can vary the memory access pattern: Either a linear or a grid-striding access is possible.
    - `reduce`: Contains the `transform_reduce` and the `reduce` function (the latter is just a convenient wrapper around the former).
        + `detail`: This contains the two possible reduce kernels.
    - `transform`: Contains two `transform` variants with one and two input iterators.
        + `detail`: This contains the transform kernel.
    - `workdiv`: This contains various working divisions for the different backends.

# Installation

## Requirements

[Alpaka](https://github.com/alpaka-group/alpaka)
Tested versions:
  * 0.5
  * dev (1d921dc0640af57c052e6fda78310561ba950e54)

## Install Vikunja

```bash
git clone https://github.com/ComputationalRadiationPhysics/vikunja.git
mkdir vikunja/build
cd vikunja/build
cmake ..
cmake --install .
```

## Enable tests

```bash
cmake .. -DBUILD_TESTING=ON
```

## Enable Examples

```bash
cmake .. -Dvikunja_BUILD_EXAMPLES=ON
```
Two small examples can be found in the folder `example/`.

# Using Vikunja in a Project

You need to import vikunja in the `CMakeLists.txt` of your Project to use it.

```
cmake_minimum_required(VERSION 3.15)

set(_TARGET_NAME example)
project(${_TARGET_NAME})

find_package(vikunja REQUIRED)

alpaka_add_executable(${_TARGET_NAME} main.cpp)
target_link_libraries(${_TARGET_NAME}
  PUBLIC
  vikunja::vikunja
)
```

Vikunja supports `find_package(vikunja)` and `add_subdirectory(/path/to/vikunja)`. Alpaka is a dependency during the configure time of your project. If it is not installed in a default location, you can use the cmake argument `-Dalpaka_DIR=/path/to/alpakaInstall/lib/cmake/alpaka` or the environment variable `ALPAKA_ROOT=/path/to/alpakaInstall/` to find it.

# Documentation

An API documentation can be generated with [doxygen](https://www.doxygen.nl/index.html). Install doxygen and run

```bash
cmake .. -Dvikunja_BUILD_DOXYGEN=ON
cmake --build . -t doc
```

The documentation can be found in `build/doc`.

# Format the code

The code is formatted with `clang-format-11`.

* Format a single file with: `clang-format -i --style=file <sourcefile>`
* If you want to format the entire code base execute the following command from alpaka’s top-level directory: `find example include test -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i --style=file`
