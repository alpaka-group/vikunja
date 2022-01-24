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
inline void benchmark_reduce(int size)
{
    std::vector<TData> hostMemInput(size);
    for(int i = 0; i < size; ++i)
    {
        hostMemInput[i] = static_cast<TData>(i) + static_cast<TData>(1);
    }

    thrust::device_vector<TData> devMemInput(hostMemInput);

    TData result = thrust::reduce(devMemInput.begin(), devMemInput.end(), static_cast<TData>(0));

    TData expected_result = (static_cast<TData>(size) * (static_cast<TData>(size) + 1)) / static_cast<TData>(2);
    // verify, that vikunja reduce is working with problem size
    REQUIRE(expected_result == Approx(result));

    // honeypot to check that the function call in the benchmark block has not been removed by the optimizer
    result = static_cast<TData>(0);

    BENCHMARK("reduce thrust")
    {
        result = thrust::reduce(devMemInput.begin(), devMemInput.end(), static_cast<TData>(0));
    };

    REQUIRE(expected_result == Approx(result));
}

TEST_CASE("bechmark reduce - int", "[reduce][thrust][int]")
{
    using Data = int;
    int size = GENERATE(100, 100'000, 1'270'000, 1'600'000);
    benchmark_reduce<Data>(size);
}

TEST_CASE("bechmark reduce - float", "[reduce][thrust][float]")
{
    using Data = float;
    int size = GENERATE(100, 100'000, 2'000'000);
    benchmark_reduce<Data>(size);
}

TEST_CASE("bechmark reduce - double", "[reduce][thrust][double]")
{
    using Data = double;
    int size = GENERATE(100, 100'000, 1'270'000, 2'000'000);
    benchmark_reduce<Data>(size);
}
