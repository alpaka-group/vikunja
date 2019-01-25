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

namespace vikunja {
namespace reduce {

    template<typename TAcc, typename TRed, typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>, typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>, typename TFunc, typename TInputIterator, typename TDevAcc, typename TDevHost, typename TQueue, typename TIdx >
    auto deviceReduce(TDevAcc &devAcc, TDevHost &devHost, TQueue &queue,  TIdx n, TInputIterator const &buffer,  TFunc const &func) -> TRed {

        // ok, now we have to think about what to do now
        if(n == 0) {
            return static_cast<TRed>(0);
        }
        constexpr uint64_t blockSize = WorkDivPolicy::template getBlockSize<TAcc>();
        using Dim = alpaka::dim::Dim<TAcc>;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, TIdx>;
        // in case n < blockSize, the block reductions only work
        // if the MemAccessPolicy maps the correct values.
        if(n < blockSize || n < 1024) {
            auto resultBuffer(alpaka::mem::buf::alloc<TRed, TIdx>(devAcc, static_cast<TIdx>(1)));
            WorkDiv dummyWorkDiv{static_cast<TIdx>(1), static_cast<TIdx>(1), static_cast<TIdx>(1)};
            detail::SmallProblemReduceKernel<TFunc> kernel;
            alpaka::kernel::exec<TAcc>(queue, dummyWorkDiv, kernel, buffer, alpaka::mem::view::getPtrNative(resultBuffer), n, func);
            TRed result;
            alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1)};
            alpaka::mem::view::copy(queue, resultView, resultBuffer, 1);
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
                          static_cast<TIdx>(1) };
        WorkDiv singleBlockWorkDiv{ static_cast<TIdx>(1),
                          static_cast<TIdx>(workDivBlockSize),
                          static_cast<TIdx>(1) };

        // allocate helper buffers
        // this should not destroy the original data
        // TODO move this to external method
        auto secondPhaseBuffer(alpaka::mem::buf::alloc<TRed, TIdx >(devAcc, gridSize));

        detail::BlockThreadReduceKernel<blockSize, MemAccessPolicy, TRed, TFunc> multiBlockKernel, singleBlockKernel;

        // execute kernels
        alpaka::kernel::exec<TAcc>(queue, multiBlockWorkDiv, multiBlockKernel, buffer, alpaka::mem::view::getPtrNative(secondPhaseBuffer), n, func);
        alpaka::kernel::exec<TAcc>(queue, singleBlockWorkDiv, singleBlockKernel, alpaka::mem::view::getPtrNative(secondPhaseBuffer), alpaka::mem::view::getPtrNative(secondPhaseBuffer), gridSize, func);

        TRed result;
        alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1)};
        alpaka::mem::view::copy(queue, resultView, secondPhaseBuffer, 1);
        // wait for result, otherwise the async CPU queue causes a segfault
        alpaka::wait::wait(queue);
        return result;
    }
}
}