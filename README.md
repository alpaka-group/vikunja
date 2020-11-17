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

## Current Alpaka version
For the tests to run, Alpaka needs to be provided as an external dependency (e.g. via cmake -DALPAKA_ROOT=...).

Supported alpaka versions

* 0.5
* dev (1d921dc0640af57c052e6fda78310561ba950e54)

## Examples
Two small examples can be found in the folder `example/`.

## Format the code

The code is formatted with `clang-format-11`.

* Format a single file with: `clang-format -i --style=file <sourcefile>`
* If you want to format the entire code base execute the following command from alpaka’s top-level directory: `find example include test -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i --style=file`
