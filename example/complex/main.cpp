/* Copyright 2021 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/reduce/reduce.hpp>
#include <vikunja/transform/transform.hpp>

#include <alpaka/alpaka.hpp>

#include <complex>
#include <iostream>
#include <vector>

int main()
{
    // Define the accelerator here. Must be one of the enabled accelerators.
    using Acc = alpaka::AccCpuSerial<alpaka::DimInt<1u>, std::uint64_t>;
    // using Acc = alpaka::AccCpuOmp2Blocks<alpaka::DimInt<1u>, std::uint64_t>;
    // using Acc = alpaka::AccGpuCudaRt<alpaka::DimInt<1u>, std::uint64_t>;

    // Type of the data that will be reduced
    using Data = float;

    // Alpaka index type
    using Idx = alpaka::Idx<Acc>;
    // Alpaka dimension type
    using Dim = alpaka::Dim<Acc>;
    // Type of the extent vector
    using Vec = alpaka::Vec<Dim, Idx>;
    // number of elements to reduce
    const Idx n = static_cast<Idx>(6400);
    // create extent
    Vec extent(Vec::all(static_cast<Idx>(n)));

    // define device, platform, and queue types.
    using DevAcc = alpaka::Dev<Acc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    // using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::Dev<Acc>>;
    using PltfHost = alpaka::PltfCpu;
    using DevHost = alpaka::Dev<PltfHost>;
    using QueueAcc = alpaka::Queue<Acc, alpaka::Blocking>;
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
    auto hInputMem1(alpaka::allocBuf<alpaka::Complex<Data>, Idx>(devHost, extent));
    auto hInputMem2(alpaka::allocBuf<alpaka::Complex<Data>, Idx>(devHost, extent));
    auto hOutputMem(alpaka::allocBuf<alpaka::Complex<Data>, Idx>(devHost, extent));
    auto dInputMem1(alpaka::allocBuf<alpaka::Complex<Data>, Idx>(devAcc, extent));
    auto dInputMem2(alpaka::allocBuf<alpaka::Complex<Data>, Idx>(devAcc, extent));
    auto dOutputMem(alpaka::allocBuf<alpaka::Complex<Data>, Idx>(devAcc, extent));

    // Fill memory on host with numbers from 0...n-1.
    alpaka::Complex<Data>* hInputMem1Ptr = alpaka::getPtrNative(hInputMem1);
    alpaka::Complex<Data>* hInputMem2Ptr = alpaka::getPtrNative(hInputMem2);
    for(Idx i = 0; i < n; ++i)
    {
        Data i_cast = static_cast<Data>(i);
        hInputMem1Ptr[i] = alpaka::Complex<Data>(i_cast, i_cast);
        hInputMem2Ptr[i] = static_cast<Data>(2) * alpaka::Complex<Data>(i_cast, i_cast);
    }

    // Copy to accelerator.
    alpaka::memcpy(queueAcc, dInputMem1, hInputMem1, extent);
    alpaka::memcpy(queueAcc, dInputMem2, hInputMem2, extent);

    // Use lambda function for transformation
    auto sub = [] ALPAKA_FN_HOST_ACC(alpaka::Complex<Data> const& a, alpaka::Complex<Data> const& b) { return a - b; };
    std::cout << "Testing accelerator: " << alpaka::getAccName<Acc>() << " with size: " << n << "\n";


    // TRANSFORM CALL:
    // Takes the arguments: accelerator device, host device, accelerator queue, size of data, input pointer-like,
    // output pointer-like, transform lambda. Can be in-place or out-of-place.
    vikunja::transform::deviceTransform<Acc>(
        devAcc,
        queueAcc,
        n,
        alpaka::getPtrNative(dInputMem2),
        alpaka::getPtrNative(dInputMem1),
        alpaka::getPtrNative(dOutputMem),
        sub);

    alpaka::memcpy(queueAcc, hOutputMem, dOutputMem, extent);

    std::vector<alpaka::Complex<Data>> expected_result_transform(
        alpaka::getPtrNative(hInputMem1),
        alpaka::getPtrNative(hInputMem1) + n);

    std::vector<alpaka::Complex<Data>> result_transform(
        alpaka::getPtrNative(hOutputMem),
        alpaka::getPtrNative(hOutputMem) + n);

    if(expected_result_transform == result_transform)
    {
        std::cout << "Transform was successful!\n";
    }
    else
    {
        std::cout << "Transform was not successful!\n";
    }

    auto transform = [] ALPAKA_FN_HOST_ACC(Acc const& acc, alpaka::Complex<Data> const& a) -> Data
    { return alpaka::math::abs(acc, a); };

    auto reduce = [] ALPAKA_FN_HOST_ACC(Data const& sum, Data const& item) { return sum + item; };

    Data result_sum = vikunja::reduce::deviceTransformReduce<Acc>(
        devAcc,
        devHost,
        queueAcc,
        n,
        alpaka::getPtrNative(dOutputMem),
        transform,
        reduce);
    Data expected_result_sum = static_cast<Data>(0);

    for(Idx i = 0; i < n; ++i)
    {
        expected_result_sum += std::abs(static_cast<std::complex<Data>>(hInputMem1Ptr[i]));
    }

    if(expected_result_sum == result_sum)
    {
        std::cout << "Reduce was successful!\n";
    }
    else
    {
        std::cout << "Reduce was not successful!\n";
    }

    return 0;
}
