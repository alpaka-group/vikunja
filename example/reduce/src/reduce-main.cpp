/* Copyright 2021 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/reduce/reduce.hpp>

#include <alpaka/alpaka.hpp>

#include <iostream>


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
    using Acc = alpaka::AccCpuSerial<alpaka::DimInt<1u>, std::uint64_t>;

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
    using Data = std::uint64_t;

    // The extent stores the problem size.
    using Vec = alpaka::Vec<Dim, Idx>;
    Vec extent(Vec::all(static_cast<Idx>(1)));
    extent[0] = static_cast<Idx>(6400);


    // Allocate memory for the device.
    auto deviceMem(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    // The memory is accessed via a pointer.
    Data* deviceNativePtr = alpaka::getPtrNative(deviceMem);
    // Allocate memory for the host.
    auto hostMem(alpaka::allocBuf<Data, Idx>(devHost, extent));
    Data* hostNativePtr = alpaka::getPtrNative(hostMem);

    // Initialize the host memory with 1 to extent.prod()+1 .
    for(Idx i = 0; i < extent.prod(); ++i)
    {
        hostNativePtr[i] = static_cast<Data>(i + 1);
    }

    // Copy data to the device.
    alpaka::memcpy(queueAcc, deviceMem, hostMem, extent);

    // Use a lambda function to define the reduction function.
    auto sum = [] ALPAKA_FN_HOST_ACC(Data const i, Data const j) { return i + j; };

    Idx reduceResult = vikunja::reduce::deviceReduce<Acc>(
        devAcc, // The device that executes the algorithm.
        devHost, // The host is necessary to allocate memory for the result.
        queueAcc, // Queue in which the algorithm is enqueued.
        extent.prod(), // Problem size
        deviceNativePtr, // Input memory
        sum // Operator
    );


    // Use a lambda function to define the transformation function.
    auto doubleNum = [] ALPAKA_FN_HOST_ACC(Data const i) { return 2 * i; };


    Idx transformReduceResult = vikunja::reduce::deviceTransformReduce<Acc>(
        devAcc, // The device that executes the algorithm.
        devHost, // The host is necessary to allocate memory for the result.
        queueAcc, // Queue in which the algorithm is enqueued.
        extent.prod(), // Problem size
        deviceNativePtr, // Input memory
        doubleNum, // transformation operator
        sum // reduction operator
    );

    std::cout << "Testing accelerator: " << alpaka::getAccName<Acc>() << " with size: " << extent.prod() << "\n";

    // Verify the reduction result.
    auto expectedReduceResult = (extent.prod() * (extent.prod() + 1) / 2);
    if(expectedReduceResult == reduceResult)
    {
        std::cout << "Reduce was successful!\n";
    }
    else
    {
        std::cout << "Reduce was not successful!\n"
                  << "expected result: " << expectedReduceResult << "\n"
                  << "actual result: " << reduceResult << std::endl;
    }

    // Verify the transform-reduction result.
    auto expectedTransformReduceResult = expectedReduceResult * 2;

    if(expectedTransformReduceResult == transformReduceResult)
    {
        std::cout << "TransformReduce was successful!\n";
    }
    else
    {
        std::cout << "TransformReduce was not successful!\n"
                  << "expected result: " << expectedTransformReduceResult << "\n"
                  << "actual result: " << transformReduceResult << std::endl;
    }

    return 0;
}
