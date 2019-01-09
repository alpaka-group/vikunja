/**
 * \file
 * Copyright 2018 Jonas Schenke, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "vikunja/reduce/detail/alpakaConfig.hpp"
#include "vikunja/reduce/detail/kernel.hpp"
#include "vikunja/GenericLambdaKernel.hpp"
#include "vikunja/reduce/reduce.hpp"
#include <alpaka/alpaka.hpp>
#include <cstdlib>
#include <iostream>
#include <chrono>

// hardcode the serial CPU accelerator
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

// use defines of a specific accelerator
using Accelerator = CpuThreads;

using DevAcc = Accelerator::DevAcc;
using DevHost = Accelerator::DevHost;
using QueueAcc = Accelerator::Stream;
using Acc = Accelerator::Acc;
using PltfAcc = Accelerator::PltfAcc;
using PltfHost = Accelerator::PltfHost;
using MaxBlockSize = Accelerator::MaxBlockSize;

int testApiReduce()
{
    constexpr Idx n = (1 << 27);
    constexpr Idx blocksPerGrid = 8;
    constexpr Idx threadsPerBlock = 1;
    constexpr Idx elementsPerThread = n / blocksPerGrid / threadsPerBlock + 1;

    DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    QueueAcc queue(devAcc);
    WorkDiv workdiv{
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread
    };

    auto deviceMem{alpaka::mem::buf::alloc<uint64_t, Idx>(devAcc, n)};
    auto hostMem{alpaka::mem::buf::alloc<uint64_t, Idx>(devHost, n)};
    alpaka::mem::view::copy(queue, deviceMem, hostMem, n);
    auto identityAssign = [](Idx i, Idx* arr) {
        arr[i] = i + 1;
    };
    auto sum = [](Idx i, Idx j) {
        return i + j;
    };
    vikunja::GenericLambdaKernel<decltype(identityAssign)> initKernel{identityAssign};
    alpaka::kernel::exec<Acc>(queue, workdiv, initKernel, n, alpaka::mem::view::getPtrNative(deviceMem));

    auto start = std::chrono::high_resolution_clock::now();
    Idx reduceResult = vikunja::reduce::deviceReduce<1, Acc>(devAcc, devHost, queue, n, deviceMem, sum, static_cast<uint64_t>(0));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::duration diff = end - start;
    auto expectedResult = (n * (n + 1) / 2);
    std::cout << "Expected: " << expectedResult << "\n";
    std::cout << "Got: " << reduceResult << "\n";
    std::cout << "Equals? " << (expectedResult == reduceResult) << "\n";
    std::cout << "Runtime? " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << "\n";
    auto rawPointer = alpaka::mem::view::getPtrNative(deviceMem);
    uint64_t sumVal  = 0;
    start = std::chrono::high_resolution_clock::now();
    for(Idx i = 0; i < n; ++i) {
        //sumVal = sum(sum(sum(sum(sumVal, rawPointer[i]), rawPointer[i+1]), rawPointer[i+2]), rawPointer[i+3]);
        sumVal = sum(sumVal, rawPointer[i]);
        //sumVal += rawPointer[i] + rawPointer[i+1] + rawPointer[i+2] + rawPointer[i+3];
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << sumVal <<"\n";
    std::cout << "Runtime? " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    return 0;
}


int main()
{
    return testApiReduce();
}

#else

int main() {
    return EXIT_SUCCESS;
}

#endif
