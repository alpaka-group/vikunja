/* Copyright 2022 Anton Reinhard
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/bench/memory.hpp>
#include <vikunja/mem/iterator/ZipIterator.hpp>
#include <vikunja/test/AlpakaSetup.hpp>
#include <vikunja/test/utility.hpp>
#include <vikunja/transform/transform.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <numeric>

#include <catch2/catch.hpp>

template<typename TIdx>
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

    using TIteratorTuplePtr = std::tuple<uint64_t*, double*>;
    using TIteratorTupleVal = std::tuple<uint64_t, double>;
    using ZipIterator = vikunja::mem::iterator::ZipIterator<TIteratorTuplePtr, TIteratorTupleVal>;

    INFO((vikunja::test::print_acc_info<typename Setup::Dim>(size)));

    Setup setup;
    Vec extent = Vec::all(static_cast<typename Setup::Idx>(size));

    auto devMemInputInt = vikunja::bench::allocate_mem_iota<uint64_t>(
        setup,
        extent,
        static_cast<uint64_t>(1), // first value
        static_cast<uint64_t>(1) // increment
    );
    auto devMemInputDouble = vikunja::bench::allocate_mem_iota<double>(
        setup,
        extent,
        static_cast<double>(1), // first value
        static_cast<double>(1) // increment
    );
    ZipIterator devMemInputPtrBegin(std::make_tuple(alpaka::getPtrNative(devMemInputInt), alpaka::getPtrNative(devMemInputDouble)));
    ZipIterator devMemInputPtrEnd = devMemInputPtrBegin + size;

    auto devMemOutput = alpaka::allocBuf<TIteratorTupleVal, typename Setup::Idx>(setup.devAcc, extent);
    TIteratorTupleVal* devMemOutputPtrBegin = alpaka::getPtrNative(devMemOutput);

    auto hostMemOutput = alpaka::allocBuf<TIteratorTupleVal, typename Setup::Idx>(setup.devHost, extent);
    TIteratorTupleVal* hostMemOutputPtrBegin = alpaka::getPtrNative(hostMemOutput);

    auto functor = [] ALPAKA_FN_HOST_ACC(TIteratorTupleVal const t) -> auto { return std::make_tuple(2 * std::get<0>(t), 2 * std::get<1>(t)); };

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
        REQUIRE(static_cast<uint64_t>(2) * static_cast<uint64_t>(i + 1) == Approx(std::get<0>(hostMemOutputPtrBegin[i])));
        REQUIRE(static_cast<double>(2) * static_cast<double>(i + 1) == Approx(std::get<1>(hostMemOutputPtrBegin[i])));
    }

    // honeypot to check that the function call in the benchmark block has not been removed by the optimizer
    hostMemOutputPtrBegin[0] = std::make_tuple(static_cast<uint64_t>(42), static_cast<double>(42));

    BENCHMARK("transform zipiter vikunja")
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

    REQUIRE(static_cast<uint64_t>(2) == Approx(std::get<0>(hostMemOutputPtrBegin[0])));
    REQUIRE(static_cast<double>(2) == Approx(std::get<1>(hostMemOutputPtrBegin[0])));
}

TEMPLATE_TEST_CASE("bechmark transform zipiter", "[benchmark][transform][vikunja][iterator]", int)
{
    using Idx = std::uint64_t;

    transform_benchmark<Idx>(GENERATE(100, 100'000, 1'270'000, 2'000'000));
}
