//
// Created by mewes30 on 19.12.18.
//

#pragma once

#include <alpaka/alpaka.hpp>
#include <type_traits>
#include "vikunja/reduce/detail/BlockThreadReduceKernel.hpp"

namespace vikunja {
namespace reduce {

    template<uint64_t blockSize, typename TAcc, typename TRed, typename TFunc, typename TBuffer, typename TDevAcc, typename TDevHost, typename TQueue, typename TIdx >
    auto deviceReduce(TDevAcc &devAcc, TDevHost &devHost, TQueue &queue,  TIdx n, TBuffer &buffer,  TFunc const &func, TRed const &init) -> TRed {

        // TODO: take care of queue management
        using Dim = alpaka::dim::Dim<TAcc>;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, TIdx>;

        // calculate workdiv sizes
        TIdx blockCount = static_cast<TIdx>(
                alpaka::acc::getAccDevProps<TAcc>(devAcc).m_multiProcessorCount *
                8);
        TIdx maxBlockCount = static_cast<TIdx>(
                (((n + 1) / 2) - 1) / static_cast<TIdx>(blockSize) + 1);

        std::cout << "blockCount: " << blockCount << "\n";
        WorkDiv multiBlockWorkDiv{ static_cast<TIdx>(blockCount),
                          static_cast<TIdx>(blockSize),
                          static_cast<TIdx>(1) };
        WorkDiv singleBlockWorkDiv{ static_cast<TIdx>(1),
                          static_cast<TIdx>(blockSize),
                          static_cast<TIdx>(1) };

        // allocate helper buffers
        // this should not destroy the original data
        auto secondPhaseBuffer{alpaka::mem::buf::alloc<TRed, TIdx >(devAcc, n)};

        detail::BlockThreadReduceKernel<blockSize, TRed, TFunc> multiBlockKernel, singleBlockKernel;

        // execute kernels
        alpaka::kernel::exec<TAcc>(queue, multiBlockWorkDiv, multiBlockKernel, alpaka::mem::view::getPtrNative(buffer), alpaka::mem::view::getPtrNative(secondPhaseBuffer), n, func);
        alpaka::kernel::exec<TAcc>(queue, singleBlockWorkDiv, singleBlockKernel, alpaka::mem::view::getPtrNative(secondPhaseBuffer), alpaka::mem::view::getPtrNative(secondPhaseBuffer), blockCount, func);
        //stub
        //alpaka::mem::view::copy(queue, secondPhaseBuffer, copyOfBuffer, blockCount);

        TRed result;
        alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1)};
        alpaka::mem::view::copy(queue, resultView, secondPhaseBuffer, 1);

        return result;
    }
}
}