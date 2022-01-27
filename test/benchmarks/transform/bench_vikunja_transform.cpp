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
#include <vikunja/transform/transform.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <catch2/catch.hpp>

template<typename TData, typename TIdx>
inline void transform_benchmark(TIdx size)
{
    using Setup = vikunja::test::TestAlpakaSetup<
        alpaka::DimInt<1u>, // dim
        TIdx, // Idx
        alpaka::AccCpuSerial, // host type
        alpaka::ExampleDefaultAcc, // device type
        alpaka::Blocking // queue type
        >;
    using Vec = alpaka::Vec<typename Setup::Dim, typename Setup::Idx>;

    INFO((vikunja::test::print_acc_info<typename Setup::Dim>(size)));

    Setup setup;
    Vec extent = Vec::all(static_cast<typename Setup::Idx>(size));

    auto devMemInput = vikunja::bench::allocate_mem_iota<TData>(
        setup,
        extent,
        static_cast<TData>(1), // first value
        static_cast<TData>(1) // increment
    );
    TData* devMemInputPtrBegin = alpaka::getPtrNative(devMemInput);
    TData* devMemInputPtrEnd = devMemInputPtrBegin + size;

    auto devMemOutput = alpaka::allocBuf<TData, typename Setup::Idx>(setup.devAcc, extent);
    TData* devMemOutputPtrBegin = alpaka::getPtrNative(devMemOutput);

    auto hostMemOutput = alpaka::allocBuf<TData, typename Setup::Idx>(setup.devHost, extent);
    TData* hostMemOutputPtrBegin = alpaka::getPtrNative(hostMemOutput);

    auto functor = [] ALPAKA_FN_HOST_ACC(TData const i) -> TData { return 2 * i; };

    vikunja::transform::deviceTransform<typename Setup::Acc>(
        setup.devAcc,
        setup.queueAcc,
        devMemInputPtrBegin,
        devMemInputPtrEnd,
        devMemOutputPtrBegin,
        functor);

    alpaka::memcpy(setup.queueAcc, hostMemOutput, devMemOutput, extent);

    for(auto i = static_cast<typename Setup::Idx>(0); i < size; ++i)
    {
        TData expected_result = static_cast<TData>(2) * static_cast<TData>(i + 1);
        REQUIRE(expected_result == Approx(hostMemOutputPtrBegin[i]));
    }

    // honeypot to check that the function call in the benchmark block has not been removed by the optimizer
    hostMemOutputPtrBegin[0] = static_cast<TData>(42);


    BENCHMARK("transform vikunja")
    {
        return vikunja::transform::deviceTransform<typename Setup::Acc>(
            setup.devAcc,
            setup.queueAcc,
            devMemInputPtrBegin,
            devMemInputPtrEnd,
            devMemOutputPtrBegin,
            functor);
    };

    alpaka::memcpy(setup.queueAcc, hostMemOutput, devMemOutput, extent);

    REQUIRE(static_cast<TData>(2) == Approx(hostMemOutputPtrBegin[0]));
}

TEMPLATE_TEST_CASE("bechmark transform", "[benchmark][transform][vikunja]", int, float, double)
{
    using Data = TestType;
    using Idx = std::uint64_t;

    transform_benchmark<Data, Idx>(GENERATE(100, 100'000, 1'270'000, 2'000'000));
}
