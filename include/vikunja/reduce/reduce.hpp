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
#define LAST_ERROR(cmd) std::cout << "In: " << cmd << " last error is: " << cudaGetErrorString(cudaGetLastError());
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
        // in case n < blockSize, the block reductions only work
        // if the MemAccessPolicy maps the correct values.
        if(n < blockSize || n < 1024) {
            auto resultBuffer(alpaka::mem::buf::alloc<TRed, TIdx>(devAcc, static_cast<TIdx>(1u)));
            WorkDiv dummyWorkDiv{static_cast<TIdx>(1u), static_cast<TIdx>(1u), static_cast<TIdx>(1u)};
            detail::SmallProblemReduceKernel kernel;
            alpaka::kernel::exec<TAcc>(queue, dummyWorkDiv, kernel, buffer, alpaka::mem::view::getPtrNative(resultBuffer), n, transformFunc, func);
            TRed result;
            alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1u)};
            alpaka::mem::view::copy(queue, resultView, resultBuffer, static_cast<TIdx>(1u));
            alpaka::wait::wait(queue);
            return result;
        }


        TIdx gridSize = WorkDivPolicy::template getGridSize<TAcc>(devAcc);

        TIdx maxGridSize = static_cast<TIdx>(
                (((n + 1) / 2) - 1) / static_cast<TIdx>(blockSize) + 1);
        if(gridSize > maxGridSize) {
            gridSize = maxGridSize;
        }

        TIdx workDivGridSize = gridSize;
        TIdx workDivBlockSize = blockSize;

        WorkDiv multiBlockWorkDiv{ static_cast<TIdx>(workDivGridSize),
                          static_cast<TIdx>(workDivBlockSize),
                          static_cast<TIdx>(1u) };
        WorkDiv singleBlockWorkDiv{ static_cast<TIdx>(1u),
                          static_cast<TIdx>(workDivBlockSize),
                          static_cast<TIdx>(1u) };

        // allocate helper buffers
        // this should not destroy the original data
        // TODO move this to external method

        LAST_ERROR("beforeInit")
        auto secondPhaseBuffer(alpaka::mem::buf::alloc<TRed, TIdx >(devAcc, gridSize));
        LAST_ERROR("afterInit")

        detail::BlockThreadReduceKernel<blockSize, MemAccessPolicy, TRed> multiBlockKernel, singleBlockKernel;

        // execute kernels
        alpaka::kernel::exec<TAcc>(queue, multiBlockWorkDiv, multiBlockKernel, buffer, alpaka::mem::view::getPtrNative(secondPhaseBuffer), n, transformFunc, func);
        alpaka::kernel::exec<TAcc>(queue, singleBlockWorkDiv, singleBlockKernel, alpaka::mem::view::getPtrNative(secondPhaseBuffer), alpaka::mem::view::getPtrNative(secondPhaseBuffer), gridSize, detail::Identity<TRed>(), func);

        TRed result;
        alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1u)};
      	alpaka::mem::view::copy(queue, resultView, secondPhaseBuffer, static_cast<TIdx>(1u));
        // wait for result, otherwise the async CPU queue causes a segfault
        alpaka::wait::wait(queue);
        return result;
    }

    template<typename TAcc, typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>, typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>, typename TFunc, typename TInputIterator, typename TDevAcc, typename TDevHost, typename TQueue, typename TIdx >
    auto deviceReduce(TDevAcc &devAcc, TDevHost &devHost, TQueue &queue,  TIdx const &n, TInputIterator const &buffer,  TFunc const &func) -> decltype(func(*buffer, *buffer)) {

        using TRed = decltype(func(*buffer, *buffer));
        return deviceTransformReduce<TAcc, WorkDivPolicy, MemAccessPolicy, detail::Identity<TRed>, TFunc, TInputIterator, TDevAcc, TDevHost, TQueue, TIdx>(devAcc, devHost, queue, n, buffer, detail::Identity<TRed>(), func);
    }
}
}
