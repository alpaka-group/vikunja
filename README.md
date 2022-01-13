# Vikunja

[![Build Status](https://gitlab.com/hzdr/crp/vikunja/badges/master/pipeline.svg)](https://gitlab.com/hzdr/crp/vikunja/-/commits/master/)
[![Documentation Status](https://readthedocs.org/projects/vikunja/badge/?version=latest)](https://vikunja.readthedocs.io)
[![Doxygen](https://img.shields.io/badge/API-Doxygen-blue.svg)](https://vikunja.readthedocs.io/en/latest/doxygen/index.html)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-orange.svg)](https://isocpp.org/)
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

## Build and Run Tests

```bash
cd vikunja/build
cmake .. -DBUILD_TESTING=ON
ctest
```

## Enable Examples

```bash
cmake .. -Dvikunja_BUILD_EXAMPLES=ON
```
Examples can be found in the folder `example/`.

# Getting Started

The following source code shows an application that uses vikunja to replace all values in a vector with their absolute values.

```c++
#include <vikunja/transform/transform.hpp>

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <iostream>
#include <random>


int main()
{
    // Define the accelerator.
    // The accelerator decides on which processor type the vikunja algorithm will be executed.
    // The accelerators must be enabled during the CMake configuration to be available.
    //
    // It is possible to choose from a set of accelerators:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccOmp5
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    using Acc = alpaka::AccCpuOmp2Blocks<alpaka::DimInt<1u>, int>;

    // Create a device that executes the algorithm.
    // For example, it can be a CPU or GPU Nr. 0 or 1 in a multi-GPU system.
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    // The host device is required if the devAcc does not use the same memory as the host.
    // For example, if the host is a CPU and the device is a GPU.
    auto const devHost(alpaka::getDevByIdx<alpaka::PltfCpu>(0u));

    // All algorithms must be enqueued so that they are executed in the correct order.
    using QueueAcc = alpaka::Queue<Acc, alpaka::Blocking>;
    QueueAcc queueAcc(devAcc);

    // Dimension of the problem. 1D in this case (inherited from the Accelerator).
    using Dim = alpaka::Dim<Acc>;
    // The index type needs to fit the problem size.
    // A smaller index type can reduce the execution time.
    // In this case the index type is inherited from the Accelerator: std::uint64_t.
    using Idx = alpaka::Idx<Acc>;
    // Type of the user data.
    using Data = int;

    // The extent stores the problem size.
    using Vec = alpaka::Vec<Dim, Idx>;
    Vec extent(Vec::all(static_cast<Idx>(10)));

    // Allocate memory for the device.
    auto deviceMem(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    // The memory is accessed via a pointer.
    Data* deviceNativePtr = alpaka::getPtrNative(deviceMem);
    // Allocate memory for the host.
    auto hostMem(alpaka::allocBuf<Data, Idx>(devHost, extent));
    Data* hostNativePtr = alpaka::getPtrNative(hostMem);

    // Initialize the host memory with random values from -10 to 10.
    std::uniform_int_distribution<Data> distribution(-10, 10);
    std::default_random_engine generator;
    std::generate(
        hostNativePtr,
        hostNativePtr + extent.prod(),
        [&distribution, &generator]() { return distribution(generator); });

    // Copy data to the device.
    alpaka::memcpy(queueAcc, deviceMem, hostMem, extent);

    // Use a lambda function to define the transformation function.
    // Returns the absolute value of each input
    auto abs = [] ALPAKA_FN_HOST_ACC(auto const& acc, Data const j) { return alpaka::math::abs(acc, j); };

    vikunja::transform::deviceTransform<Acc>(
        devAcc, // The device that executes the algorithm.
        queueAcc, // Queue in which the algorithm is enqueued.
        extent.prod(), // Problem size
        deviceNativePtr, // Input memory
        deviceNativePtr, // Operator
        abs);

    // Copy data to the host.
    alpaka::memcpy(queueAcc, hostMem, deviceMem, extent);

    for(Data i = 0; i < extent.prod(); ++i)
    {
        std::cout << hostNativePtr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(vikunjaAbs)

add_subdirectory(vikunja REQUIRED)

alpaka_add_executable(${CMAKE_PROJECT_NAME} main.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE vikunja::vikunja)
```

Build instructions:
```bash
# the source folder contains the main.cpp and the CMakeLists.txt
cd <folder/with/source/code>
mkdir build && cd build
# configure build with OpenMP backend enabled
cmake .. -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=ON
# compile application
cmake --build .
# run application
./vikunjaAbs # output: 10 8 5 1 1 6 10 4 4 9
```

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
