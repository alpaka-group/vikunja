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
    // number of elements to reduce
    const Idx n = static_cast<Idx>(6400);

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

    // Use Lambda function for reduction
    auto sum = [] ALPAKA_FN_HOST_ACC(TRed const i, TRed const j) { return i + j; };
    auto doubleNum = [] ALPAKA_FN_HOST_ACC(TRed const i) { return 2 * i; };
    std::cout << "Testing accelerator: " << alpaka::getAccName<TAcc>() << " with size: " << n << "\n"
              << "Testing constant iterator with value: 10\n";

    // Create the constant iterator
    vikunja::mem::iterator::ConstantIterator<TRed> constantIter(10);

    // REDUCE CALL:
    // Takes the arguments: accelerator device, host device, accelerator queue, size of data, pointer-like to memory,
    // reduce lambda.
    Idx reduceResult = vikunja::reduce::deviceReduce<TAcc>(
        devAcc, 
        devHost, 
        queueAcc, 
        n, 
        constantIter,
        sum);

    // check reduce result
    auto expectedResult = n * 10;
    std::cout << "Expected reduce result: " << expectedResult << ", real result: " << reduceResult << "\n";

    // TRANSFORM_REDUCE CALL:
    // Takes the arguments: accelerator device, host device, accelerator queue, size of data, pointer-like to memory,
    // transform lambda, reduce lambda.
    Idx transformReduceResult = vikunja::reduce::deviceTransformReduce<TAcc>(
        devAcc,
        devHost,
        queueAcc,
        n,
        constantIter,
        doubleNum,
        sum);

    // check transform result
    auto expectedTransformReduce = expectedResult * 2;
    std::cout << "Expected transform_reduce result: " << expectedTransformReduce
              << ", real result: " << transformReduceResult << "\n";

    return 0;
}
