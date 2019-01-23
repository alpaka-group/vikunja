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

    template<typename TAcc,typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>, typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>, typename TRed, typename TFunc, typename TBuffer, typename TDevAcc, typename TDevHost, typename TQueue, typename TIdx >
    auto deviceReduce(TDevAcc &devAcc, TDevHost &devHost, TQueue &queue,  TIdx n, TBuffer &buffer,  TFunc const &func, TRed const &init) -> TRed {

        if(n == 0) return init;
        constexpr uint64_t blockSize = WorkDivPolicy::template getBlockSize<TAcc>();
        using Dim = alpaka::dim::Dim<TAcc>;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, TIdx>;
        // in case n < blockSize, the block reductions only work
        // if the MemAccessPolicy maps the correct values.
        if(n < blockSize || n < 1024) {
            auto resultBuffer(alpaka::mem::buf::alloc<TRed, TIdx>(devAcc, static_cast<TIdx>(1)));
            WorkDiv dummyWorkDiv{static_cast<TIdx>(1), static_cast<TIdx>(1), static_cast<TIdx>(1)};
            detail::SmallProblemReduceKernel<TFunc> kernel;
            alpaka::kernel::exec<TAcc>(queue, dummyWorkDiv, kernel, alpaka::mem::view::getPtrNative(buffer), alpaka::mem::view::getPtrNative(resultBuffer), n, func);
            TRed result;
            alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1)};
            alpaka::mem::view::copy(queue, resultView, resultBuffer, 1);
            alpaka::wait::wait(queue);
            return func(result, init);
        }


        TIdx gridSize = WorkDivPolicy::template getGridSize<TAcc>(devAcc);
        // calculate workdiv sizes
        /*TIdx gridSize = static_cast<TIdx>(
                alpaka::acc::getAccDevProps<TAcc>(devAcc).m_multiProcessorCount *
                8);*/
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
        alpaka::kernel::exec<TAcc>(queue, multiBlockWorkDiv, multiBlockKernel, alpaka::mem::view::getPtrNative(buffer), alpaka::mem::view::getPtrNative(secondPhaseBuffer), n, func);
        std::cout << "After first reduce\n";
        alpaka::kernel::exec<TAcc>(queue, singleBlockWorkDiv, singleBlockKernel, alpaka::mem::view::getPtrNative(secondPhaseBuffer), alpaka::mem::view::getPtrNative(secondPhaseBuffer), gridSize, func);

        TRed result;
        alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1)};
        alpaka::mem::view::copy(queue, resultView, secondPhaseBuffer, 1);
        // wait for result, otherwise the async CPU queue causes a segfault
        alpaka::wait::wait(queue);
        return func(result, init);
    }
}
}