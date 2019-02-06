//
// Created by hauke on 25.01.19.
//

#pragma once

#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>
#include <vikunja/workdiv/BlockBasedWorkDiv.hpp>
#include <vikunja/transform/detail/BlockThreadTransformKernel.hpp>
#include <alpaka/alpaka.hpp>
#include <type_traits>

namespace vikunja {
    namespace transform {


        template<typename TAcc, typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>, typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>, typename TFunc, typename TInputIterator, typename TOutputIterator, typename TDevAcc, typename TQueue, typename TIdx >
        auto deviceTransform(TDevAcc &devAcc, TQueue &queue,  TIdx const &n, TInputIterator const &source, TOutputIterator const &destination, TFunc const &func) -> void {
            if(n == 0) {
                return;
            }
            constexpr uint64_t blockSize = WorkDivPolicy::template getBlockSize<TAcc>();
            using Dim = alpaka::dim::Dim<TAcc>;
            using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, TIdx>;
            if(n < blockSize) {
                // TODO fix this?
                // maybe not needed
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
            detail::BlockThreadTransformKernel<blockSize, MemAccessPolicy> kernel;
            alpaka::kernel::exec<TAcc>(queue, multiBlockWorkDiv, kernel, source, destination, n, func);
        }
    } // transform
} // vikunja
