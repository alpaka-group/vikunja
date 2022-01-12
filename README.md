# Vikunja

[![Build Status](https://gitlab.com/hzdr/crp/vikunja/badges/master/pipeline.svg)](https://gitlab.com/hzdr/crp/vikunja/-/commits/master/)
[![Documentation Status](https://readthedocs.org/projects/vikunja/badge/?version=latest)](https://vikunja.readthedocs.io)
[![Doxygen](https://img.shields.io/badge/API-Doxygen-blue.svg)](https://vikunja.readthedocs.io/en/latest/doxygen/index.html)
[![Language](https://img.shields.io/badge/language-C%2B%2B14-orange.svg)](https://isocpp.org/)
[![Platforms](https://img.shields.io/badge/platform-linux-lightgrey.svg)](https://github.com/alpaka-group/vikunja)
[![License](https://img.shields.io/badge/license-MPL--2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)

![vikunja](docs/source/logo/vikunja_logo.png)

Vikunja is a performance portable algorithm library that defines functions operating on ranges of elements for a variety of purposes . It supports the execution on multi-core CPUs and various GPUs.

Vikunja uses [alpaka](https://github.com/alpaka-group/alpaka) to implement platform-independent primitives such as `reduce` or `transform`. 

# Installation
## Install Alpaka

Alpaka requires a [boost installation](https://github.com/alpaka-group/alpaka#dependencies).

```bash
git clone --depth 1 --branch 0.8.0 https://github.com/alpaka-group/alpaka.git
mkdir alpaka/build
cd alpaka/build
cmake ..
cmake --install .
```

For more information see the [alpaka GitHub](https://github.com/alpaka-group/alpaka) repository. It is recommended to use the latest release version. Vikunja supports `alpaka` from version `0.6` up to version `0.8`.

## Install Vikunja

```bash
git clone https://github.com/alpaka-group/vikunja.git
mkdir vikunja/build
cd vikunja/build
cmake ..
cmake --install .
```

# Build and Run Tests

```bash
cd vikunja/build
cmake .. -DBUILD_TESTING=ON
ctest
```

# Enable Examples

```bash
cmake .. -Dvikunja_BUILD_EXAMPLES=ON
```
Examples can be found in the folder `example/`.

# Documentation

- You can find the general documentation here: https://vikunja.readthedocs.io/en/latest/
- You can find the API documentation here: https://vikunja.readthedocs.io/en/latest/doxygen/index.html

# Authors

## Maintainers* and Core Developers

- Simeon Ehrig*

## Former Members, Contributions and Thanks

- Dr. Michael Bussmann
- Hauke Mewes
- René Widera
- Bernhard Manfred Gruber
- Jan Stephan
- Dr. Jiří Vyskočil
- Matthias Werner