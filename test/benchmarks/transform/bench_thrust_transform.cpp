/* Copyright 2022 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vector>

#include <catch2/catch.hpp>
#include <thrust/device_vector.h>

template<typename TData>
inline void transform_benchmark(int size)
{
    std::vector<TData> hostMemInput(size);
    for(int i = 0; i < size; ++i)
    {
        hostMemInput[i] = static_cast<TData>(i) + static_cast<TData>(1);
    }

    thrust::device_vector<TData> devMemInput(hostMemInput);
    thrust::device_vector<TData> devMemOutput(size);

    auto functor = [] __device__(TData const i) -> TData { return 2 * i; };
    thrust::transform(devMemInput.begin(), devMemInput.end(), devMemOutput.begin(), functor);

    std::vector<TData> hostMemOutput(size);
    thrust::copy(devMemOutput.begin(), devMemOutput.end(), hostMemOutput.begin());

    for(int i = 0; i < size; ++i)
    {
        TData expected_result = static_cast<TData>(2) * static_cast<TData>(i + 1);
        REQUIRE(expected_result == Approx(hostMemOutput[i]));
    }
    // honeypot to check that the function call in the benchmark block has not been removed by the optimizer
    hostMemOutput[0] = static_cast<TData>(42);

    BENCHMARK("transform thrust")
    {
        return thrust::transform(devMemInput.begin(), devMemInput.end(), devMemOutput.begin(), functor);
    };

    thrust::copy(devMemOutput.begin(), devMemOutput.end(), hostMemOutput.begin());

    REQUIRE(static_cast<TData>(2) == Approx(hostMemOutput[0]));
}

TEMPLATE_TEST_CASE("bechmark transform", "[benchmark][thrust][vikunja]", int, float, double)
{
    using Data = TestType;

    transform_benchmark<Data>(GENERATE(100, 100'000, 1'270'000, 2'000'000));
}
