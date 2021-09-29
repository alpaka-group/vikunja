/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vector>
#include <numeric>

#include <iostream>
#include "cuplaVikReduce.hpp"

int main()
{
    constexpr int size = 10;

    std::vector<int> input(size);
    std::iota(input.begin(), input.end(), 0);

    int expected_sum = std::reduce(input.begin(), input.end());

    int sum = reduce(std::move(input));

    if(expected_sum == sum)
    {
        return 0;
    }
    else
    {
        std::cout << "error" << std::endl;
        std::cout << "expected sum: " << expected_sum << std::endl;
        std::cout << "sum: " << sum << std::endl;
        return 1;
    }
}
