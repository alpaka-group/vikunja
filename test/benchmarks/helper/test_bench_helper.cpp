/* Copyright 2022 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/bench/memory.hpp>
#include <vikunja/test/AlpakaSetup.hpp>
#include <vikunja/test/utility.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <catch2/catch.hpp>


TEMPLATE_TEST_CASE("allocate_mem_iota compare std::iota", "[iota]", int, float, double)
{
    using Data = TestType;
    using Setup = vikunja::test::TestAlpakaSetup<
        alpaka::DimInt<1u>, // dim
        int, // Idx
        alpaka::AccCpuSerial, // host type
        alpaka::ExampleDefaultAcc, // device type
        alpaka::Blocking // queue type
        >;
    using Vec = alpaka::Vec<Setup::Dim, Setup::Idx>;

    Setup::Idx size = GENERATE(1, 10, 3045, 2'000'000);
    Data begin = GENERATE(0, 1, 45, -42);

    INFO((vikunja::test::print_acc_info<Setup::Dim>(size)));
    INFO("begin: " + std::to_string(begin));

    Setup setup;
    Vec extent = Vec::all(static_cast<Setup::Idx>(size));

    auto devMem = vikunja::bench::allocate_mem_iota<Data>(setup, extent, begin);
    auto hostMem(alpaka::allocBuf<Data, typename Setup::Idx>(setup.devHost, extent));
    Data* const hostMemPtr(alpaka::getPtrNative(hostMem));

    alpaka::memcpy(setup.queueAcc, hostMem, devMem, extent);

    std::vector<Data> expected_result(size);
    std::iota(std::begin(expected_result), std::end(expected_result), begin);

    for(Setup::Idx i = 0; i < size; ++i)
    {
        REQUIRE(static_cast<Data>(expected_result[i]) == hostMemPtr[i]);
    }
}

TEMPLATE_TEST_CASE("allocate_mem_iota different increment", "[iota]", int, float, double)
{
    using Data = TestType;
    using Setup = vikunja::test::TestAlpakaSetup<
        alpaka::DimInt<1u>, // dim
        int, // Idx
        alpaka::AccCpuSerial, // host type
        alpaka::ExampleDefaultAcc, // device type
        alpaka::Blocking // queue type
        >;
    using Vec = alpaka::Vec<Setup::Dim, Setup::Idx>;

    Setup::Idx size = GENERATE(1, 10, 3045);
    Data begin = GENERATE(0, 1, 45, -42);
    Data increment = GENERATE(1, -1, 45, -42);

    INFO((vikunja::test::print_acc_info<Setup::Dim>(size)));
    INFO("begin: " + std::to_string(begin));
    INFO("increment: " + std::to_string(increment));

    Setup setup;
    Vec extent = Vec::all(static_cast<Setup::Idx>(size));

    auto devMem = vikunja::bench::allocate_mem_iota<Data>(setup, extent, begin, increment);
    auto hostMem(alpaka::allocBuf<Data, typename Setup::Idx>(setup.devHost, extent));
    Data* const hostMemPtr(alpaka::getPtrNative(hostMem));

    alpaka::memcpy(setup.queueAcc, hostMem, devMem, extent);

    for(Setup::Idx i = 0; i < size; ++i)
    {
        Data expected_result = begin + static_cast<Data>(i) * increment;
        REQUIRE_MESSAGE(expected_result == hostMemPtr[i], "failed with index: " + std::to_string(i));
    }
}

TEMPLATE_TEST_CASE("allocate_mem_constant", "[iota]", int, float, double)
{
    using Data = TestType;
    using Setup = vikunja::test::TestAlpakaSetup<
        alpaka::DimInt<1u>, // dim
        int, // Idx
        alpaka::AccCpuSerial, // host type
        alpaka::ExampleDefaultAcc, // device type
        alpaka::Blocking // queue type
        >;
    using Vec = alpaka::Vec<Setup::Dim, Setup::Idx>;

    Setup::Idx size = GENERATE(1, 10, 3045, 2'000'000);
    Data constant = GENERATE(0, 1, 45, -42);

    INFO((vikunja::test::print_acc_info<Setup::Dim>(size)));
    INFO("constant: " + std::to_string(constant));

    Setup setup;
    Vec extent = Vec::all(static_cast<Setup::Idx>(size));

    auto devMem = vikunja::bench::allocate_mem_constant<Data>(setup, extent, constant);
    auto hostMem(alpaka::allocBuf<Data, typename Setup::Idx>(setup.devHost, extent));
    Data* const hostMemPtr(alpaka::getPtrNative(hostMem));

    alpaka::memcpy(setup.queueAcc, hostMem, devMem, extent);

    for(Setup::Idx i = 0; i < size; ++i)
    {
        REQUIRE(static_cast<Data>(constant) == hostMemPtr[i]);
    }
}
