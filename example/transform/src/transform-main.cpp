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
    // Define the accelerator here. Must be one of the enabled accelerators.
    using TAcc = alpaka::AccCpuSerial<alpaka::DimInt<3u>, std::uint64_t>;

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
    const Idx n = static_cast<Idx>(6400);
    // create extent
    Vec extent(Vec::all(static_cast<Idx>(1)));
    extent[xIndex] = n;

    // define device, platform, and queue types.
    using DevAcc = alpaka::Dev<TAcc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    // using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::Dev<TAcc>>;
    using PltfHost = alpaka::PltfCpu;
    using DevHost = alpaka::Dev<PltfHost>;
    using QueueAcc = alpaka::Queue<TAcc, alpaka::Blocking>;
    using QueueHost = alpaka::QueueCpuBlocking;

    // Get the host device.
    DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    // Get a queue on the host device.
    QueueHost queueHost(devHost);
    // Select a device to execute on.
    DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    // Get a queue on the accelerator device.
    QueueAcc queueAcc(devAcc);

    // allocate memory both on host and device.
    auto deviceMem(alpaka::allocBuf<TRed, Idx>(devAcc, extent));
    auto hostMem(alpaka::allocBuf<TRed, Idx>(devHost, extent));
    // Fill memory on host with numbers from 0...n-1.
    TRed* hostNative = alpaka::getPtrNative(hostMem);
    for(Idx i = 0; i < n; ++i)
    {
        // std::cout << i << "\n";
        hostNative[i] = static_cast<TRed>(i + 1);
    }
    // Copy to accelerator.
    alpaka::memcpy(queueAcc, deviceMem, hostMem, extent);
    // Use lambda function for transformation
    auto doubleNum = [] ALPAKA_FN_HOST_ACC(TRed const& i) { return 2 * i; };
    std::cout << "Testing accelerator: " << alpaka::getAccName<TAcc>() << " with size: " << n << "\n";


    // TRANSFORM CALL:
    // Takes the arguments: accelerator device, host device, accelerator queue, size of data, input pointer-like,
    // output pointer-like, transform lambda. Can be in-place or out-of-place.
    vikunja::transform::deviceTransform<TAcc>(
        devAcc,
        queueAcc,
        n,
        alpaka::getPtrNative(deviceMem),
        alpaka::getPtrNative(deviceMem),
        doubleNum);

    // copy back to host to validate
    alpaka::memcpy(queueAcc, hostMem, deviceMem, extent);
    TRed resultSum = 0;
    for(Idx i = 0; i < n; ++i)
    {
        resultSum += hostNative[i];
    }
    TRed expectedResult = (n * (n + 1));
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
