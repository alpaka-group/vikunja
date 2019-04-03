//
// Created by mewes30 on 19.12.18.
//

#pragma once

#include <alpaka/alpaka.hpp>
#include <type_traits>
#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>
#include <vikunja/workdiv/BlockBasedWorkDiv.hpp>
#include <vikunja/reduce/detail/SmallProblemReduceKernel.hpp>
#include <vikunja/reduce/detail/BlockThreadReduceKernel.hpp>
#include <iostream>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define LAST_ERROR(cmd) {cudaDeviceSynchronize();std::cout << "In: " << cmd << " last error is: " << cudaGetErrorString(cudaGetLastError()) << "\n";}
#else
#define LAST_ERROR(cmd)
#endif

namespace vikunja {
namespace reduce {
    namespace detail {
        template <typename T>
        struct Identity {
            constexpr ALPAKA_FN_HOST_ACC T operator()(T const &arg) const {
                return arg;
            }
        };
    }

    template<typename TAcc, typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>, typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>, typename TTransformFunc, typename TFunc, typename TInputIterator, typename TDevAcc, typename TDevHost, typename TQueue, typename TIdx >
    auto deviceTransformReduce(TDevAcc &devAcc, TDevHost &devHost, TQueue &queue,  TIdx const &n, TInputIterator const &buffer, TTransformFunc const &transformFunc, TFunc const &func) -> decltype(func(transformFunc(*buffer), transformFunc(*buffer))) {

        // TODO: more elegant way to obtain return type + avoid that double declaration
        using TRed = decltype(func(transformFunc(*buffer), transformFunc(*buffer)));
        // ok, now we have to think about what to do now
        // TODO: think of proper solution for this.
        if(n == 0) {
//            return static_cast<TRed>(0);
        }
        constexpr uint64_t blockSize = WorkDivPolicy::template getBlockSize<TAcc>();
        using Dim = alpaka::dim::Dim<TAcc>;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, TIdx>;
        using Vec = alpaka::vec::Vec<Dim, TIdx>;
        constexpr TIdx xIndex = Dim::value - 1u;

        Vec elementsPerThread(Vec::all(static_cast<TIdx>(1u)));
        Vec threadsPerBlock(Vec::all(static_cast<TIdx>(1u)));
        Vec blocksPerGrid(Vec::all(static_cast<TIdx>(1u)));

        Vec const resultBufferExtent(Vec::all(static_cast<TIdx>(1u)));
        auto resultBuffer(alpaka::mem::buf::alloc<TRed, TIdx>(devAcc, resultBufferExtent));

        // in case n < blockSize, the block reductions only work
        // if the MemAccessPolicy maps the correct values.
        if(n < blockSize || n < 1024) {
            auto resultBuffer(alpaka::mem::buf::alloc<TRed, TIdx>(devAcc, resultBufferExtent));
            WorkDiv dummyWorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
            detail::SmallProblemReduceKernel kernel;
            alpaka::kernel::exec<TAcc>(queue, dummyWorkDiv, kernel, buffer, alpaka::mem::view::getPtrNative(resultBuffer), n, transformFunc, func);
            auto resultView(alpaka::mem::buf::alloc<TRed, TIdx >(devHost, resultBufferExtent));
           // TRed result;
           // alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1u)};
            alpaka::mem::view::copy(queue, resultView, resultBuffer, resultBufferExtent);
            alpaka::wait::wait(queue);
            auto result = alpaka::mem::view::getPtrNative(resultView);
            return result[0];
        }


        TIdx gridSize = WorkDivPolicy::template getGridSize<TAcc>(devAcc);

        TIdx maxGridSize = static_cast<TIdx>(
                (((n + 1) / 2) - 1) / static_cast<TIdx>(blockSize) + 1);
        if(gridSize > maxGridSize) {
            gridSize = maxGridSize;
        }

        TIdx workDivGridSize = gridSize;
        TIdx workDivBlockSize = blockSize;

        blocksPerGrid[xIndex] = workDivGridSize;
        threadsPerBlock[xIndex] = workDivBlockSize;

        Vec const singleElementsPerThread(Vec::all(static_cast<TIdx>(1u)));
        Vec singleThreadsPerBlock(Vec::all(static_cast<TIdx>(1u)));
        Vec const singleBlocksPerGrid(Vec::all(static_cast<TIdx>(1u)));
        singleThreadsPerBlock[xIndex] = workDivBlockSize;

        Vec sharedMemExtent(Vec::all(static_cast<TIdx>(1u)));
        sharedMemExtent[xIndex] = gridSize;

        //std::cout << "elementsPerThread: " << singleElementsPerThread << "\n";
        //std::cout << "threadsPerBlock: " << singleThreadsPerBlock << "\n";
        //std::cout << "blocksPerGrid: " << singleBlocksPerGrid << "\n";

        WorkDiv multiBlockWorkDiv{ blocksPerGrid, threadsPerBlock, elementsPerThread };
        WorkDiv singleBlockWorkDiv{ singleBlocksPerGrid, singleThreadsPerBlock, singleElementsPerThread};

        // allocate helper buffers
        // this should not destroy the original data
        // TODO move this to external method

        //LAST_ERROR("beforeInit")
        auto secondPhaseBuffer(alpaka::mem::buf::alloc<TRed, TIdx >(devAcc, sharedMemExtent));
        //LAST_ERROR("afterInit")

        detail::BlockThreadReduceKernel<blockSize, MemAccessPolicy, TRed> multiBlockKernel, singleBlockKernel;
        //std::cout << "after kernel create\n";
        // execute kernels
        alpaka::kernel::exec<TAcc>(queue, multiBlockWorkDiv, multiBlockKernel, buffer, alpaka::mem::view::getPtrNative(secondPhaseBuffer), n, transformFunc, func);
        //std::cout << "after first kernel\n";
        alpaka::kernel::exec<TAcc>(queue, singleBlockWorkDiv, singleBlockKernel, alpaka::mem::view::getPtrNative(secondPhaseBuffer), alpaka::mem::view::getPtrNative(secondPhaseBuffer), gridSize, detail::Identity<TRed>(), func);
        //LAST_ERROR("afterLaunch");
        //std::cout << "after second kernel\n";
        auto sharedMemPointer = alpaka::mem::view::getPtrNative(secondPhaseBuffer);

        //TRed result;
        auto resultView(alpaka::mem::buf::alloc<TRed, TIdx >(devHost, resultBufferExtent));
        //alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1u)};
        //std::cout << "after view setup\n";
        alpaka::mem::view::copy(queue, resultView, secondPhaseBuffer, resultBufferExtent);
        //LAST_ERROR("afterCopy")
        //std::cout << "after view copy\n";
        // wait for result, otherwise the async CPU queue causes a segfault
        alpaka::wait::wait(queue);
        //std::cout << "after wait \n";

        std::cout << "before fetch result";
        auto result = alpaka::mem::view::getPtrNative(resultView);
        std::cout << "after fetch result";
        return result[0];
    }

    template<typename TAcc, typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>, typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>, typename TFunc, typename TInputIterator, typename TDevAcc, typename TDevHost, typename TQueue, typename TIdx >
    auto deviceReduce(TDevAcc &devAcc, TDevHost &devHost, TQueue &queue,  TIdx const &n, TInputIterator const &buffer,  TFunc const &func) -> decltype(func(*buffer, *buffer)) {

        using TRed = decltype(func(*buffer, *buffer));
        return deviceTransformReduce<TAcc, WorkDivPolicy, MemAccessPolicy, detail::Identity<TRed>, TFunc, TInputIterator, TDevAcc, TDevHost, TQueue, TIdx>(devAcc, devHost, queue, n, buffer, detail::Identity<TRed>(), func);
    }
}
}
