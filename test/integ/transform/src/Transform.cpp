/* Copyright 2019-2021 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/test/AlpakaSetup.hpp>
#include <vikunja/transform/transform.hpp>
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <catch2/catch.hpp>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <cstdio>
#include <vector>
#include <thread>

#if defined(VIKUNJA_TRANSFORM_COMPARING_BENCHMARKS) && defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#    include <thrust/device_vector.h>
//#include <thrust/reduce.h>
#    include <thrust/functional.h>
#endif
struct incOne
{
    template<typename TAcc>
    ALPAKA_FN_HOST_ACC std::uint64_t operator()(TAcc const&, const std::uint64_t& val)
    {
        return val + 1;
    }
};

template<typename TAcc>
struct TestAlpakaEnv
{
public:
    using Idx = alpaka::Idx<TAcc>;
    using Dim = alpaka::Dim<TAcc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using DevAcc = alpaka::Dev<TAcc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using PltfHost = alpaka::PltfCpu;
    using DevHost = alpaka::Dev<PltfHost>;
    using QueueAcc = alpaka::Queue<DevAcc, alpaka::Blocking>;

    static constexpr Idx xIndex = Dim::value - 1u;

    DevAcc acc;
    DevHost host;
    QueueAcc queue;

    TestAlpakaEnv() : acc(alpaka::getDevByIdx<PltfAcc>(0u)), host(alpaka::getDevByIdx<PltfHost>(0u)), queue(acc)
    {
    }

    Vec getSingleDimensionVec(Idx const val)
    {
        Vec vec(Vec::all(static_cast<Idx>(1)));
        vec[xIndex] = val;
        return vec;
    }

    template<typename TValue>
    auto allocHost(Vec const& extent)
    {
        return alpaka::allocBuf<TValue, Idx>(host, extent);
    }
    template<typename TValue>
    auto allocAcc(Vec const& extent)
    {
        return alpaka::allocBuf<TValue, Idx>(acc, extent);
    }

    template<typename TLambda>
    auto fillDeviceBuffer(Vec const& extent, TLambda const& lambda)
    {
        using TValue = decltype(lambda(static_cast<Idx>(0u)));
        auto deviceMem(allocAcc<TValue>(extent));
        auto hostMem(allocHost<TValue>(extent));
        auto hostPtr = alpaka::getPtrNative(hostMem);
        const auto n = extent[xIndex];
        for(Idx i = 0; i < n; ++i)
        {
            hostPtr[i] = lambda(i);
        }
        alpaka::memcpy(queue, deviceMem, hostMem, extent);
        return deviceMem;
    }
};


struct TestTemplate
{
private:
    const uint64_t memSize;

public:
    TestTemplate(uint64_t const memSize) : memSize(memSize)
    {
    }

    template<typename TAcc>
    void operator()()
    {
        using TRed = uint64_t;

        /*using Idx = alpaka::Idx<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        const Idx n = static_cast<Idx>(memSize);
        constexpr Idx blocksPerGrid = 8;
        constexpr Idx threadsPerBlock = 1;
        const Idx elementsPerThread = n / blocksPerGrid / threadsPerBlock + 1;

        using Vec = alpaka::Vec<Dim, Idx>;
        constexpr Idx xIndex = Dim::value - 1u;

        Vec extent(Vec::all(static_cast<Idx>(1)));
        extent[xIndex] = n;

        using DevAcc = alpaka::Dev<TAcc>;
        using PltfAcc = alpaka::Pltf<DevAcc>;
        // Async queue makes things slower on CPU?
        // using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::Dev<TAcc>>;
        using PltfHost = alpaka::PltfCpu;
        using DevHost = alpaka::Dev<PltfHost>;
        using QueueAcc = //alpaka::QueueCpuAsync;
        typename std::conditional<std::is_same<PltfAcc, alpaka::PltfCpu>::value, alpaka::QueueCpuBlocking,
#ifdef  ALPAKA_ACC_GPU_CUDA_ENABLED
                alpaka::QueueCudaRtBlocking
#else
                alpaka::QueueCpuBlocking
#endif
        >::type;*/
        TestAlpakaEnv<TAcc> testAlpakaEnv;
        using Idx = typename TestAlpakaEnv<TAcc>::Idx;
        const Idx n = memSize;
        // auto &devHost = testAlpakaEnv.host;
        auto& devAcc = testAlpakaEnv.acc;
        auto& queueAcc = testAlpakaEnv.queue;
        auto extent = testAlpakaEnv.getSingleDimensionVec(memSize);

        auto incrementOne = [=] ALPAKA_FN_HOST_ACC(TAcc const&, Idx i) { return i + 1; };
        auto deviceMem(testAlpakaEnv.fillDeviceBuffer(extent, [=](Idx i) { return i + 1; }));
        auto hostMem(testAlpakaEnv.template allocHost<TRed>(extent));

        std::cout << "Testing accelerator: " << alpaka::getAccName<TAcc>() << " with size: " << n << "\n";

        auto start = std::chrono::high_resolution_clock::now();
        // insert new call here
        vikunja::transform::deviceTransform<TAcc>(
            devAcc,
            queueAcc,
            n,
            alpaka::getPtrNative(deviceMem),
            alpaka::getPtrNative(deviceMem),
            incrementOne);
        auto end = std::chrono::high_resolution_clock::now();
        alpaka::memcpy(queueAcc, hostMem, deviceMem, extent);
        auto nativePointer = alpaka::getPtrNative(hostMem);
        bool isValid = true;
        for(Idx i = 0; i < n; ++i)
        {
            isValid = isValid && nativePointer[i] == (i + 2);
        }
        REQUIRE(isValid);
        std::cout << "Runtime of " << alpaka::getAccName<TAcc>() << ": "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
    }
};

TEST_CASE("Test transform", "[transform]")
{
    std::cout << "Test transform" << std::endl;
    using TestAccs = alpaka::ExampleDefaultAcc<alpaka::DimInt<3u>, std::uint64_t>;
    // std::cout << std::thread::hardware_concurrency() << "\n";
    SECTION("deviceTransform")
    {
        std::vector<uint64_t> memorySizes{1, 10, 16, 777, (1 << 10) + 1, 1 << 15, 1 << 25, 1 << 27};

        for(auto& memSize : memorySizes)
        {
            // alpaka currently offers no function to keep all active back-ends
            // alpaka::meta::forEachType<TestAccs>(TestTemplate(memSize));
            TestTemplate t(memSize);
            t.operator()<TestAccs>();
        }
#ifdef VIKUNJA_TRANSFORM_COMPARING_BENCHMARKS
        std::cout << "---------------------------------------------\n";
        std::cout << "Now performing some benchmarks...\n";
        const std::uint64_t size = memorySizes.back();
        std::vector<uint64_t> reduce(size);
        for(uint64_t i = 0; i < reduce.size(); ++i)
        {
            reduce[i] = i + 1;
        }
        auto start = std::chrono::high_resolution_clock::now();
        for(uint64_t i = 0; i < reduce.size(); ++i)
        {
            ++reduce[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Runtime of dumb: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
        uint64_t tSum = 0;
        for(uint64_t i = 0; i < reduce.size(); ++i)
        {
            tSum += reduce[i];
        }
        std::cout << "New sum: " << (tSum - (size * (size + 1) / 2)) << "\n";
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // test against thrust
        thrust::device_vector<std::uint64_t> deviceReduce(reduce);
        start = std::chrono::high_resolution_clock::now();
        thrust::transform(deviceReduce.begin(), deviceReduce.end(), deviceReduce.begin(), incOne());
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        tSum = thrust::reduce(
            deviceReduce.begin(),
            deviceReduce.end(),
            static_cast<std::uint64_t>(0),
            thrust::plus<std::uint64_t>());
        std::cout << "Runtime of thrust reduce: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
        std::cout << "tSum = " << tSum << "\n";


#    endif

#endif
    }
}

template<typename TAcc>
void testTemplateDoubleInput(const uint64_t memSize)
{
    using TRed = uint64_t;

    TestAlpakaEnv<TAcc> testAlpakaEnv;
    using Idx = typename TestAlpakaEnv<TAcc>::Idx;
    const Idx n = memSize;

    auto& devAcc = testAlpakaEnv.acc;
    auto& queueAcc = testAlpakaEnv.queue;
    auto extent = testAlpakaEnv.getSingleDimensionVec(memSize);

    auto deviceMemInput1(testAlpakaEnv.fillDeviceBuffer(extent, [=](Idx i) { return i + 1; }));
    auto deviceMemInput2(testAlpakaEnv.fillDeviceBuffer(extent, [=](Idx i) { return i * 2; }));
    auto deviceMemOutput(testAlpakaEnv.fillDeviceBuffer(extent, [=](Idx) { return static_cast<Idx>(0); }));
    auto hostMem(testAlpakaEnv.template allocHost<TRed>(extent));

    std::cout << "Testing accelerator: " << alpaka::getAccName<TAcc>() << " with size: " << n << "\n";

    auto transfromFunc = [=] ALPAKA_FN_HOST_ACC(TAcc const&, Idx i1, Idx i2) { return i2 - i1; };

    auto start = std::chrono::high_resolution_clock::now();
    vikunja::transform::deviceTransform<TAcc>(
        devAcc,
        queueAcc,
        n,
        alpaka::getPtrNative(deviceMemInput1),
        alpaka::getPtrNative(deviceMemInput2),
        alpaka::getPtrNative(deviceMemOutput),
        transfromFunc);
    auto end = std::chrono::high_resolution_clock::now();

    alpaka::memcpy(queueAcc, hostMem, deviceMemOutput, extent);

    auto nativePointer = alpaka::getPtrNative(hostMem);
    bool isValid = true;
    for(Idx i = 0; i < n; ++i)
    {
        isValid = isValid && nativePointer[i] == ((i * 2) - (i + 1));
    }

    REQUIRE(isValid);
    std::cout << "Runtime of " << alpaka::getAccName<TAcc>() << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
}

TEST_CASE("Test transform two input iterators", "[transform]")
{
    std::cout << "Test transform two input iterators" << std::endl;
    using TestAccs = alpaka::ExampleDefaultAcc<alpaka::DimInt<3u>, std::uint64_t>;
    std::vector<uint64_t> memorySizes{1, 10, 16, 777, (1 << 10) + 1, 1 << 15, 1 << 25, 1 << 27};

    for(auto& memSize : memorySizes)
    {
        testTemplateDoubleInput<TestAccs>(memSize);
    }
}
