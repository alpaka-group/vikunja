/* Copyright 2021 Simeon Ehrig
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

#include <numeric>

#include <catch2/catch.hpp>

template<typename TData>
inline void transform_benchmark(int size)
{
    using Setup = vikunja::test::TestAlpakaSetup<
        alpaka::DimInt<1u>, // dim
        std::uint64_t, // Idx
        alpaka::AccCpuSerial, // host type
        alpaka::ExampleDefaultAcc, // device type
        alpaka::Blocking // queue type
        >;
    using Vec = alpaka::Vec<Setup::Dim, Setup::Idx>;

    INFO((vikunja::test::print_acc_info<Setup::Dim>(size)));

    Setup setup;
    Vec extent = Vec::all(static_cast<Setup::Idx>(size));

    auto devMemInput = vikunja::bench::allocate_mem_iota<TData>(
        setup,
        extent,
        static_cast<TData>(1), // first value
        static_cast<TData>(1) // increment
    );
    TData* devMemInputPtrBegin = alpaka::getPtrNative(devMemInput);
    TData* devMemInputPtrEnd = devMemInputPtrBegin + size;

    auto devMemOutput = alpaka::allocBuf<TData, Setup::Idx>(setup.devAcc, extent);
    TData* devMemOutputPtrBegin = alpaka::getPtrNative(devMemOutput);

    auto hostMemOutput = alpaka::allocBuf<TData, Setup::Idx>(setup.devHost, extent);
    TData* hostMemOutputPtrBegin = alpaka::getPtrNative(hostMemOutput);
    TData* hostMemOutputPtrEnd = hostMemOutputPtrBegin + size;

    auto functor = [] ALPAKA_FN_HOST_ACC(TData const i) -> TData { return 2 * i; };

    vikunja::transform::deviceTransform<Setup::Acc>(
        setup.devAcc,
        setup.queueAcc,
        devMemInputPtrBegin,
        devMemInputPtrEnd,
        devMemOutputPtrBegin,
        functor);

    alpaka::memcpy(setup.queueAcc, hostMemOutput, devMemOutput, extent);

    TData result = std::reduce(hostMemOutputPtrBegin, hostMemOutputPtrEnd, static_cast<TData>(0));
    TData expected_result = extent.prod() * (extent.prod() + 1);

    // verify, that vikunja transform is working with problem size
    REQUIRE(expected_result == Approx(result));

    // honeypot to check that the function call in the benchmark block has not been removed by the optimizer
    hostMemOutputPtrBegin[0] = static_cast<TData>(42);


    BENCHMARK("transform vikunja")
    {
        vikunja::transform::deviceTransform<Setup::Acc>(
            setup.devAcc,
            setup.queueAcc,
            devMemInputPtrBegin,
            devMemInputPtrEnd,
            devMemOutputPtrBegin,
            functor);
    };

    alpaka::memcpy(setup.queueAcc, hostMemOutput, devMemOutput, extent);

    result = std::reduce(hostMemOutputPtrBegin, hostMemOutputPtrEnd, static_cast<TData>(0));
    REQUIRE(expected_result == Approx(result));
}

TEST_CASE("benchmark transform - int", "[transform][vikunja][int]")
{
    using Data = int;
    int size = GENERATE(100, 100'000, 1'270'000, 2'000'000);

    transform_benchmark<Data>(size);
}

TEST_CASE("benchmark transform - float", "[transform][vikunja][float]")
{
    using Data = float;
    // removed 1'270'000 because of rounding errors.
    int size = GENERATE(100, 100'000, 2'000'000);

    transform_benchmark<Data>(size);
}

TEST_CASE("benchmark transform - double", "[transform][vikunja][double]")
{
    using Data = double;
    int size = GENERATE(100, 100'000, 1'270'000, 2'000'000);

    transform_benchmark<Data>(size);
}
