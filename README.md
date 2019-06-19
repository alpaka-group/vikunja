# Primitives for Alpaka
This contains a `reduce` and a `transform` primitive for Alpaka, both header-only.
Directories:
* `include/vikunja`: The actual code
    - `mem/iterator`: Contains a base iterator class and a policy based iterator that can vary the memory access pattern: Either a linear or a grid-striding access is possible.
    - `reduce`: Contains the `transform_reduce` and the `reduce` function (the latter is just a convenient wrapper around the former).
        + `detail`: This contains the two possible reduce kernels.
    - `transform`: Contains two `transform` variants with one and two input iterators.
        + `detail`: This contains the transform kernel.
    - `workdiv`: This contains various working divisions for the different backends. These are currently flawed and need to be optimized, see issue #8.
    