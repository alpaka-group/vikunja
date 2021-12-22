/* Copyright 2021 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/transform/transform.hpp>

#include <alpaka/alpaka.hpp>

#include <iostream>

int main()
{
    // Define the accelerator.
    // The accelerates decides on which processor type the vikunja algorithm will be executed.
    // The accelerators must be enabled in the CMake build to be available.
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
    using Acc = alpaka::AccCpuSerial<alpaka::DimInt<1u>, std::uint64_t>;

    // Create a device that executes the algorithm.
    // For example, it can be a CPU or GPU Nr. 0 or 1 in a multi-GPU system.
    auto const devAcc(alpaka::getDevByIdx<Acc>(0u));
    // The host device is required if the devAcc does not use the same memory as the host.
    // For example, if the host is a CPU and the device is a GPU.
    auto const devHost(alpaka::getDevByIdx<alpaka::PltfCpu>(0u));

    // All algorithms must be queued so that they are executed in the correct order.
    using QueueAcc = alpaka::Queue<Acc, alpaka::Blocking>;
    QueueAcc queueAcc(devAcc);


    // Dimension of the problem.
    using Dim = alpaka::Dim<Acc>;
    // The index type needs to fit the problem size.
    // A smaller index type can reduce the execution time.
    using Idx = alpaka::Idx<Acc>;
    // Type of the user data.
    using Data = uint64_t;

    // The extends stores the problem size.
    using Vec = alpaka::Vec<Dim, Idx>;
    Vec extent(Vec::all(static_cast<Idx>(1)));
    extent[0] = static_cast<Idx>(6400);


    // Allocates memory for the device.
    auto deviceMem(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    // The memory is accessed via a pointer.
    Data* deviceNativePtr = alpaka::getPtrNative(deviceMem);
    // Allocates memory for the host.
    auto hostMem(alpaka::allocBuf<Data, Idx>(devHost, extent));
    Data* hostNativePtr = alpaka::getPtrNative(hostMem);


    // Initialize the host memory with 1 to extent.prod()+1 .
    for(Idx i = 0; i < extent.prod(); ++i)
    {
        hostNativePtr[i] = static_cast<Data>(i + 1);
    }

    // Copy data to the device.
    alpaka::memcpy(queueAcc, deviceMem, hostMem, extent);

    // Use a lambda function to define the transformation function.
    auto doubleNum = [] ALPAKA_FN_HOST_ACC(Data const& i) { return 2 * i; };

    vikunja::transform::deviceTransform<Acc>(
        devAcc, // The device that executes the algorithm.
        queueAcc, // Queue in which the algorithm enqueues.
        extent[Dim::value - 1u], // Problem size
        deviceNativePtr, // Input memory
        deviceNativePtr, // Output memory
        doubleNum // Operator
    );

    // Copy the data back to the host for validation.
    alpaka::memcpy(queueAcc, hostMem, deviceMem, extent);

    Data resultSum = std::accumulate(hostNativePtr, hostNativePtr + extent.prod(), 0);

    Data expectedResult = (extent.prod() * (extent.prod() + 1));

    std::cout << "Testing accelerator: " << alpaka::getAccName<Acc>() << " with size: " << extent.prod() << "\n";
    if(expectedResult == resultSum)
    {
        std::cout << "Transform was successful!\n";
    }
    else
    {
        std::cout << "Transform was not successful!\n";
    }

    return 0;
}
