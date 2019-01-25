//
// Created by hauke on 11.01.19.
//

#include <vikunja/test/AlpakaSetup.hpp>
#include <vikunja/GenericLambdaKernel.hpp>
#include <vikunja/transform/transform.hpp>
#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <catch2/catch.hpp>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <cstdio>
#include <vector>
#include <thread>

#if defined(VIKUNJA_TRANSFORM_COMPARING_BENCHMARKS) && defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <thrust/device_vector.h>
//#include <thrust/reduce.h>
#include <thrust/functional.h>
#endif
struct incOne {
    ALPAKA_FN_HOST_ACC std::uint64_t operator()(const std::uint64_t &val) {
        return val + 1;
    }
};


struct TestTemplate {
private:
    const uint64_t memSize;
public:

    TestTemplate(uint64_t const memSize) : memSize(memSize) {}

    template<typename TAcc>
    void operator()() {
        using Idx = alpaka::idx::Idx<TAcc>;
        using Dim = alpaka::dim::Dim<TAcc>;
        const Idx n = static_cast<Idx>(memSize);
        constexpr Idx blocksPerGrid = 8;
        constexpr Idx threadsPerBlock = 1;
        const Idx elementsPerThread = n / blocksPerGrid / threadsPerBlock + 1;

        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        // Async queue makes things slower on CPU?
       // using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::dev::Dev<TAcc>>;
        using PltfHost = alpaka::pltf::PltfCpu;
        using DevHost = alpaka::dev::Dev<PltfHost>;
        using QueueAcc = //alpaka::queue::QueueCpuAsync;
                typename std::conditional<std::is_same<PltfAcc, alpaka::pltf::PltfCpu>::value, alpaka::queue::QueueCpuSync,
#ifdef  ALPAKA_ACC_GPU_CUDA_ENABLED
        alpaka::queue::QueueCudaRtSync
#else
        alpaka::queue::QueueCpuSync
#endif
        >::type;
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
        //alpaka::mem::view::copy(queueAcc, deviceMem, hostMem, n);
        alpaka::wait::wait(queueHost);
        auto identityAssign = [=] ALPAKA_FN_HOST_ACC (Idx i, Idx* arr) {
            arr[i] = i + 1;
        };
        auto incrementOne = [=] ALPAKA_FN_HOST_ACC (Idx i) {
            return i + 1;
        };
        vikunja::GenericLambdaKernel<decltype(identityAssign)> initKernel{identityAssign};
        alpaka::kernel::exec<TAcc>(queueAcc, workdiv, initKernel, n, alpaka::mem::view::getPtrNative(deviceMem));

        std::cout << "Testing accelerator: " << alpaka::acc::getAccName<TAcc>() << " with size: " << n <<"\n";

        auto start = std::chrono::high_resolution_clock::now();
        // insert new call here
        vikunja::transform::deviceTransform<TAcc, Idx>(devAcc, devHost, queueAcc, n, alpaka::mem::view::getPtrNative(deviceMem),
                alpaka::mem::view::getPtrNative(deviceMem), incrementOne);
        auto end = std::chrono::high_resolution_clock::now();
        alpaka::mem::view::copy(queueAcc, hostMem, deviceMem, n);
        auto nativePointer = alpaka::mem::view::getPtrNative(hostMem);
        bool isValid = true;
        for(Idx i = 0; i < n; ++i) {
            isValid = isValid && nativePointer[i] == (i + 2);
        }
        REQUIRE(isValid);
        std::cout << "Runtime of " << alpaka::acc::getAccName<TAcc>() << ": "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
    }
};

TEST_CASE("Test reduce", "[reduce]")
{

    using TestAccs = alpaka::test::acc::EnabledAccs<
            alpaka::dim::DimInt<1u>,
            std::uint64_t>;
    //std::cout << std::thread::hardware_concurrency() << "\n";
    SECTION("deviceReduce") {

        std::vector<uint64_t> memorySizes{1,10, 16,  777,(1<< 10) + 1, 1 << 15, 1 << 25, 1 << 27};

        for(auto &memSize: memorySizes) {
            alpaka::meta::forEachType<TestAccs>(TestTemplate(memSize));
        }
#ifdef VIKUNJA_TRANSFORM_COMPARING_BENCHMARKS
        std::cout << "---------------------------------------------\n";
        std::cout << "Now performing some benchmarks...\n";
        const std::uint64_t size = memorySizes.back();
        std::vector<uint64_t> reduce(size);
        for(uint64_t i = 0; i < reduce.size(); ++i) {
            reduce[i] = i + 1;
        }
        auto start = std::chrono::high_resolution_clock::now();
        for(uint64_t i = 0; i < reduce.size(); ++i) {
            ++reduce[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Runtime of dump: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
        uint64_t tSum = 0;
        for(uint64_t i = 0; i < reduce.size(); ++i) {
            tSum += reduce[i];
        }
        std::cout << "New sum: " << (tSum - (size * (size + 1) / 2)) << "\n";
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // test against thrust
        thrust::device_vector<std::uint64_t> deviceReduce(reduce);
        start = std::chrono::high_resolution_clock::now();
        thrust::transform(deviceReduce.begin(), deviceReduce.end(), deviceReduce.begin(), incOne());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Runtime of thrust reduce: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
        //std::cout << "tSum = " << tSum << "\n";

#endif

#endif
    }
}