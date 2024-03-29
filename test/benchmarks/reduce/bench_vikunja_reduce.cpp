/* Copyright 2022 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/bench/memory.hpp>
#include <vikunja/reduce/reduce.hpp>
#include <vikunja/test/AlpakaSetup.hpp>
#include <vikunja/test/utility.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <numeric>

#include <catch2/catch.hpp>

template<typename TData, typename TIdx>
inline void reduce_benchmark(TIdx size)
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

    auto functor = [] ALPAKA_FN_HOST_ACC(TData const i, TData const j) -> TData { return i + j; };

    TData result = vikunja::reduce::deviceReduce<typename Setup::Acc>(
        setup.devAcc,
        setup.devHost,
        setup.queueAcc,
        devMemInputPtrBegin,
        devMemInputPtrEnd,
        functor);

    TData expected_result = (extent.prod() * (extent.prod() + static_cast<TData>(1)) / static_cast<TData>(2));

    // verify, that vikunja reduce is working with problem size
    REQUIRE(expected_result == Approx(result));

    // honeypot to check that the function call in the benchmark block has not been removed by the optimizer
    result = static_cast<TData>(0);

    BENCHMARK("reduce vikunja")
    {
        return result = vikunja::reduce::deviceReduce<typename Setup::Acc>(
                   setup.devAcc,
                   setup.devHost,
                   setup.queueAcc,
                   devMemInputPtrBegin,
                   devMemInputPtrEnd,
                   functor);
    };

    REQUIRE(expected_result == Approx(result));
}

TEMPLATE_TEST_CASE("bechmark reduce", "[benchmark][reduce][vikunja]", int, float, double)
{
    using Data = TestType;
    using Idx = std::uint64_t;

    if constexpr(std::is_same_v<Data, int>)
    {
        reduce_benchmark<Data, Idx>(GENERATE(100, 100'000, 1'270'000, 1'600'000));
    }
    else if constexpr(std::is_same_v<Data, float>)
    {
        // removed 1'270'000 because of precision errors.
        reduce_benchmark<Data, Idx>(GENERATE(100, 100'000, 2'000'000));
    }
    else if constexpr(std::is_same_v<Data, double>)
    {
        reduce_benchmark<Data, Idx>(GENERATE(100, 100'000, 1'270'000, 2'000'000));
    }
}
