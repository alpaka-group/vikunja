/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "cuplaVikReduce.hpp"

#include <vikunja/reduce/reduce.hpp>

#include <alpaka/alpaka.hpp>

#include <iostream>

#include <cupla.hpp>

int reduce(std::vector<int> const&& input)
{
    cuplaError_t err = cuplaSuccess;
    auto const size = sizeof(int) * input.size();

    cupla::AccHost const devHost(cupla::manager::Device<cupla::AccHost>::get().current());
    cupla::AccDev const devAcc(cupla::manager::Device<cupla::AccDev>::get().current());
    cupla::AccStream devQueue(cupla::manager::Stream<cupla::AccDev, cupla::AccStream>::get().stream(0));

    int* devMem = nullptr;

    err = cuplaMalloc((void**) &devMem, size);
    if(err != cuplaSuccess)
    {
        std::cerr << "cuplaMalloc failed" << std::endl;
        return -1;
    }

    err = cuplaMemcpy(devMem, input.data(), size, cuplaMemcpyHostToDevice);
    if(err != cuplaSuccess)
    {
        std::cerr << "cuplaMemcpy failed" << std::endl;
        return -1;
    }

    auto const sum = [] ALPAKA_FN_HOST_ACC(cupla::Acc const&, int const i, int const j) { return i + j; };

    int const result = vikunja::reduce::deviceReduce<cupla::Acc>(
        devAcc,
        devHost,
        devQueue,
        static_cast<cupla::IdxType>(input.size()),
        devMem,
        sum);
    return result;
}
