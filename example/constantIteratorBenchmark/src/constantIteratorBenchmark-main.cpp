/* Copyright 2021 Hauke Mewes, Simeon Ehrig, Victor
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/reduce/reduce.hpp>
#include <vikunja/mem/iterator/ConstantIterator.hpp>

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <iostream>


int main()
{
    // Define the accelerator here. Must be one of the enabled accelerators.
    using TAcc = alpaka::AccGpuCudaRt<alpaka::DimInt<3u>, std::uint64_t>;

    // Type of the data that will be reduced
    using TRed = uint64_t;

    // Alpaka index type
    using Idx = alpaka::Idx<TAcc>;
    // Alpaka dimension type
    using Dim = alpaka::Dim<TAcc>;
    // Type of the extent vector
    using Vec = alpaka::Vec<Dim, Idx>;
    // Find the index of the CUDA blockIdx.x component. Alpaka somehow reverses
    // these, i.e. the x component of cuda is always the last value in the vector
    constexpr Idx xIndex = Dim::value - 1u;
    // number of elements to reduce
    const Idx n = static_cast<Idx>(1000000);
    // create extent
    Vec extent(Vec::all(static_cast<Idx>(1)));
    extent[xIndex] = n;

    // Value for constant iterator
    const TRed constantIterVal = 10;
    // Number of benchmark iteration
    const int benchmarkIterations = 10000;

    // define device, platform, and queue types.
    using DevAcc = alpaka::Dev<TAcc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    // using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::Dev<TAcc>>;
    using PltfHost = alpaka::PltfCpu;
    using DevHost = alpaka::Dev<PltfHost>;
    using QueueAcc = alpaka::Queue<TAcc, alpaka::Blocking>;

    // Get the host device.
    DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    // Select a device to execute on.
    DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    // Get a queue on the accelerator device.
    QueueAcc queueAcc(devAcc);

    // allocate memory both on host and device.
    auto deviceMem(alpaka::allocBuf<TRed, Idx>(devAcc, extent));
    auto hostMem(alpaka::allocBuf<TRed, Idx>(devHost, extent));
    // Fill memory on host with constantIterVal.
    TRed* hostNative = alpaka::getPtrNative(hostMem);
    for(Idx i = 0; i < n; ++i)
    {
        // std::cout << i << "\n";
        hostNative[i] = static_cast<TRed>(constantIterVal);
    }
    // Copy to accelerator.
    alpaka::memcpy(queueAcc, deviceMem, hostMem, extent);

    // Use Lambda function for reduction
    auto sum = [] ALPAKA_FN_HOST_ACC(TRed const i, TRed const j) { return i + j; };
    auto doubleNum = [] ALPAKA_FN_HOST_ACC(TRed const i) { return 2 * i; };
    std::cout << "\nTesting accelerator: " << alpaka::getAccName<TAcc>() << " with size: " << n << "\n"
              << "Number of benchmark iterations: " << benchmarkIterations << "\n"
              << "=====\n"
              << "[[ Testing without constant iterator ]]\n";

    // Use chrono library to measure time;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
 
    // REDUCE CALL:
    // Takes the arguments: accelerator device, host device, accelerator queue, size of data, pointer-like to memory,
    // reduce lambda.
    // Run this once as a warm-up
    Idx reduceResult = vikunja::reduce::deviceReduce<TAcc>(
        devAcc, 
        devHost, 
        queueAcc, 
        n, 
        alpaka::getPtrNative(deviceMem),
        sum);

    // Start timer
    auto timerStart = high_resolution_clock::now();

    for (int i = 0; i < benchmarkIterations; ++i)
    {
        // REDUCE CALL:
        reduceResult = vikunja::reduce::deviceReduce<TAcc>(
            devAcc, 
            devHost, 
            queueAcc, 
            n, 
            alpaka::getPtrNative(deviceMem),
            sum);
    }

    // End timer and count duration
    auto timerEnd = high_resolution_clock::now();
    duration<double, std::milli> msDouble = timerEnd - timerStart;

    // check reduce result
    auto expectedResult = n * constantIterVal;
    std::cout << "Expected reduce result: " << expectedResult
              << ", real result: " << reduceResult << "\n"
              << "Duration: " << msDouble.count() / benchmarkIterations << "ms\n";

    // TRANSFORM_REDUCE CALL:
    // Takes the arguments: accelerator device, host device, accelerator queue, size of data, pointer-like to memory,
    // transform lambda, reduce lambda.
    // Run this once as a warm-up
    Idx transformReduceResult = vikunja::reduce::deviceTransformReduce<TAcc>(
        devAcc,
        devHost,
        queueAcc,
        n,
        alpaka::getPtrNative(deviceMem),
        doubleNum,
        sum);

    // Start timer
    timerStart = high_resolution_clock::now();

    for (int i = 0; i < benchmarkIterations; ++i)
    {
        // TRANSFORM_REDUCE CALL:
        transformReduceResult = vikunja::reduce::deviceTransformReduce<TAcc>(
            devAcc,
            devHost,
            queueAcc,
            n,
            alpaka::getPtrNative(deviceMem),
            doubleNum,
            sum);
    }

    // End timer and count duration
    timerEnd = high_resolution_clock::now();
    msDouble = timerEnd - timerStart;

    // check transform result
    auto expectedTransformReduce = expectedResult * 2;
    std::cout << "Expected transform_reduce result: " << expectedTransformReduce
              << ", real result: " << transformReduceResult << "\n"
              << "Duration: " << msDouble.count() / benchmarkIterations << "ms\n"
              << "-----\n"
              << "[[ Testing constant iterator with value: " << constantIterVal << " ]]\n";
    
    // Create the constant iterator
    vikunja::mem::iterator::ConstantIterator<TRed> constantIter(constantIterVal);

    // REDUCE CALL:
    // Run this once as a warm-up
    reduceResult = vikunja::reduce::deviceReduce<TAcc>(
        devAcc, 
        devHost, 
        queueAcc, 
        n, 
        constantIter,
        sum);

    // Start timer
    timerStart = high_resolution_clock::now();

    for (int i = 0; i < benchmarkIterations; ++i)
    {
        // REDUCE CALL:
        reduceResult = vikunja::reduce::deviceReduce<TAcc>(
            devAcc, 
            devHost, 
            queueAcc, 
            n, 
            constantIter,
            sum);
    }

    // End timer and count duration
    timerEnd = high_resolution_clock::now();
    msDouble = timerEnd - timerStart;

    // check reduce result
    expectedResult = n * constantIterVal;
    std::cout << "Expected reduce result: " << expectedResult
              << ", real result: " << reduceResult << "\n"
              << "Duration: " << msDouble.count() / benchmarkIterations << "ms\n";

    // TRANSFORM_REDUCE CALL:
    // Run this once as a warm-up
    transformReduceResult = vikunja::reduce::deviceTransformReduce<TAcc>(
        devAcc,
        devHost,
        queueAcc,
        n,
        constantIter,
        doubleNum,
        sum);

    // Start timer
    timerStart = high_resolution_clock::now();

    for (int i = 0; i < benchmarkIterations; ++i)
    {
        // TRANSFORM_REDUCE CALL:
        transformReduceResult = vikunja::reduce::deviceTransformReduce<TAcc>(
            devAcc,
            devHost,
            queueAcc,
            n,
            constantIter,
            doubleNum,
            sum);
    }

    // End timer and count duration
    timerEnd = high_resolution_clock::now();
    msDouble = timerEnd - timerStart;

    // check transform result
    expectedTransformReduce = expectedResult * 2;
    std::cout << "Expected transform_reduce result: " << expectedTransformReduce
              << ", real result: " << transformReduceResult << "\n"
              << "Duration: " << msDouble.count() / benchmarkIterations << "ms\n\n";

    return 0;
}
