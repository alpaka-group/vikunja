/* Copyright 2022 Anton Reinhard
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/mem/iterator/ConstantIterator.hpp>
#include <vikunja/reduce/reduce.hpp>
#include <vikunja/test/utility.hpp>
#include <vikunja/transform/transform.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>

#include <catch2/catch.hpp>

using TAcc = alpaka::AccCpuSerial<alpaka::DimInt<3u>, std::uint64_t>;

// Type of the data that will be transformed
using TTrans = std::uint64_t;

// Alpaka index type
using Idx = alpaka::Idx<TAcc>;
// Alpaka dimension type
using Dim = alpaka::Dim<TAcc>;

// define device, platform, and queue types.
using DevAcc = alpaka::Dev<TAcc>;
using PltfAcc = alpaka::Pltf<DevAcc>;
// using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::Dev<TAcc>>;
using PltfHost = alpaka::PltfCpu;
using DevHost = alpaka::Dev<PltfHost>;
using QueueAcc = alpaka::Queue<TAcc, alpaka::Blocking>;

struct IncOne
{
    template<typename TData>
    ALPAKA_FN_HOST_ACC TData operator()(TData const val) const
    {
        return val + 1;
    }
};

struct TimesTwo
{
    template<typename TData>
    ALPAKA_FN_HOST_ACC TData operator()(TData const val) const
    {
        return val * 2;
    }
};

TEST_CASE("ConstantIteratorTest", "[transform][iterator][lambda]")
{
    auto size = GENERATE(1, 10, 777, 1 << 10);
    auto constantIterVal = GENERATE(1, 5, 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    // Get the host device.
    DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    // Select a device to execute on.
    DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    // Get a queue on the accelerator device.
    QueueAcc queueAcc(devAcc);

    // allocate output memory both on host and device.
    auto deviceOutMem(alpaka::allocBuf<TTrans, Idx>(devAcc, size));
    auto hostOutMem(alpaka::allocBuf<TTrans, Idx>(devHost, size));

    vikunja::mem::iterator::ConstantIterator c_begin(constantIterVal);

    vikunja::transform::deviceTransform(
        devAcc,
        queueAcc,
        size,
        c_begin,
        &deviceOutMem,
        [](TTrans v) -> TTrans { return 2 * v; });

    TTrans* hostNative = alpaka::getPtrNative(hostOutMem);
    for(Idx i = 0; i < size; ++i)
    {
        REQUIRE(2 * constantIterVal == hostNative[i]);
    }
}
