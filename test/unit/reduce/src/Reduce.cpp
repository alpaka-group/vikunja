//
// Created by hauke on 11.01.19.
//

#include <vikunja/test/AlpakaSetup.hpp>
#include <vikunja/reduce/detail/BlockThreadReduceKernel.hpp>
#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <catch2/catch.hpp>
#include <cstdlib>
#include <iostream>
#include <vikunja/GenericLambdaKernel.hpp>
#include <vikunja/reduce/reduce.hpp>
#include <cstdio>

struct TestTemplate {

    template<typename TAcc>
    void operator()() {
        using Idx = alpaka::idx::Idx<TAcc>;
        using Dim = alpaka::dim::Dim<TAcc>;
        constexpr Idx n = (1 << 27);
        constexpr Idx blocksPerGrid = 8;
        constexpr Idx threadsPerBlock = 1;
        constexpr Idx elementsPerThread = n / blocksPerGrid / threadsPerBlock + 1;

        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::dev::Dev<TAcc>>;
        using PltfHost = alpaka::pltf::PltfCpu;
        using DevHost = alpaka::dev::Dev<PltfHost>;
        using QueueHost = alpaka::queue::QueueCpuSync;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
        // Get the host device.
        DevHost devHost(
                alpaka::pltf::getDevByIdx<PltfHost>(0u));
        // Get a queue on the host device.
        QueueHost queueHost(
                devHost);
        // Select a device to execute on.
        DevAcc devAcc(
                alpaka::pltf::getDevByIdx<PltfAcc>(0u));
        // Get a queue on the accelerator device.
        QueueAcc queueAcc(
                devAcc);
        WorkDiv workdiv{
                blocksPerGrid,
                threadsPerBlock,
                elementsPerThread
        };

        auto deviceMem(alpaka::mem::buf::alloc<uint64_t, Idx>(devAcc, n));
        auto hostMem(alpaka::mem::buf::alloc<uint64_t, Idx>(devHost, n));

        alpaka::mem::view::copy(queueAcc, deviceMem, hostMem, n);
        auto identityAssign = [](Idx i, Idx* arr) {
            arr[i] = i + 1;
        };
        auto sum = [](Idx i, Idx j) {
            return i + j;
        };
        vikunja::GenericLambdaKernel<decltype(identityAssign)> initKernel{identityAssign};
        alpaka::kernel::exec<TAcc>(queueAcc, workdiv, initKernel, n, alpaka::mem::view::getPtrNative(deviceMem));

        std::cout << "Testing accelerator: " << alpaka::acc::getAccName<TAcc>() << "\n";

        auto start = std::chrono::high_resolution_clock::now();
        Idx reduceResult = vikunja::reduce::deviceReduce<1, TAcc>(devAcc, devHost, queueAcc, n, deviceMem, sum, static_cast<uint64_t>(0));
        auto end = std::chrono::high_resolution_clock::now();
        auto expectedResult = (n * (n + 1) / 2);
        REQUIRE(expectedResult == reduceResult);
        std::cout << "Runtime of " << alpaka::acc::getAccName<TAcc>() << ": "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
        using MemAccess = vikunja::mem::iterator::MemAccessPolicy<TAcc>;
        std::cout << "MemAccessPolicy: " << MemAccess::getName() << "\n";
    }
};

TEST_CASE("Test reduce", "[reduce]")
{

    using TestAccs = alpaka::test::acc::EnabledAccs<
            alpaka::dim::DimInt<1u>,
            std::uint64_t>;
    SECTION("deviceReduce") {
        alpaka::meta::forEachType<TestAccs>(TestTemplate());
    }
}