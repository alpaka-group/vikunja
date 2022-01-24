/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <numeric>
#include <vector>

#include <catch2/catch.hpp>
#include <thrust/device_vector.h>

template<typename TData>
inline void benchmark_transform(int size)
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

    TData result = std::reduce(hostMemOutput.begin(), hostMemOutput.end(), static_cast<TData>(0));
    TData expected_result = static_cast<TData>(size) * (static_cast<TData>(size) + 1);
    // verify, that vikunja transform is working with problem size
    REQUIRE(expected_result == Approx(result));

    // honeypot to check that the function call in the benchmark block has not been removed by the optimizer
    hostMemOutput[0] = static_cast<TData>(42);

    BENCHMARK("transform thrust")
    {
        thrust::transform(devMemInput.begin(), devMemInput.end(), devMemOutput.begin(), functor);
    };

    thrust::copy(devMemOutput.begin(), devMemOutput.end(), hostMemOutput.begin());

    result = std::reduce(hostMemOutput.begin(), hostMemOutput.end(), static_cast<TData>(0));
    REQUIRE(expected_result == Approx(result));
}

TEST_CASE("bechmark transform - int", "[transform][thrust][int]")
{
    using Data = int;
    int size = GENERATE(100, 100'000, 1'270'000, 2'000'000);
    benchmark_transform<Data>(size);
}

TEST_CASE("bechmark transform - float", "[transform][thrust][float]")
{
    using Data = float;
    int size = GENERATE(100, 100'000, 2'000'000);
    benchmark_transform<Data>(size);
}

TEST_CASE("bechmark transform - double", "[transform][thrust][double]")
{
    using Data = double;
    int size = GENERATE(100, 100'000, 1'270'000, 2'000'000);
    benchmark_transform<Data>(size);
}
