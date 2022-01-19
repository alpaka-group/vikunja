/* Copyright 2022 Anton Reinhard
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/mem/iterator/ConstantIterator.hpp>
#include <vikunja/test/AlpakaSetup.hpp>
#include <vikunja/test/utility.hpp>
#include <vikunja/transform/transform.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>

#include <catch2/catch.hpp>

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
    // Type of the data that will be transformed
    using TTrans = std::uint64_t;

    using Setup = vikunja::test::TestAlpakaSetup<
        alpaka::DimInt<1u>, // dim
        std::uint64_t, // Idx
        alpaka::AccCpuSerial, // host type
        alpaka::ExampleDefaultAcc, // device type
        alpaka::Blocking // queue type
        >;
    using Vec = alpaka::Vec<Setup::Dim, Setup::Idx>;

    // Testmatrix: test each combination of size and constantIterVal
    Setup::Idx size = GENERATE(1, 10, 777, 1 << 10);
    auto constantIterVal = GENERATE(1, 5, 10);

    INFO((vikunja::test::print_acc_info<Setup::Dim>(size)));
    Setup setup;

    Vec extent = Vec::all(static_cast<Setup::Idx>(size));

    // allocate output memory both on host and device.
    auto deviceOutMem(alpaka::allocBuf<TTrans, Setup::Idx>(setup.devAcc, extent));
    auto hostOutMem(alpaka::allocBuf<TTrans, Setup::Idx>(setup.devHost, extent));

    vikunja::mem::iterator::ConstantIterator c_begin(constantIterVal);

    vikunja::transform::deviceTransform<Setup::Acc>(
        setup.devAcc,
        setup.queueAcc,
        size,
        c_begin,
        alpaka::getPtrNative(deviceOutMem),
        [] ALPAKA_FN_HOST_ACC(TTrans v) -> TTrans { return 2 * v; });

    alpaka::memcpy(setup.queueAcc, hostOutMem, deviceOutMem, extent);

    TTrans* hostNative = alpaka::getPtrNative(hostOutMem);
    for(Setup::Idx i = 0; i < size; ++i)
    {
        REQUIRE(2 * constantIterVal == hostNative[i]);
    }
}
