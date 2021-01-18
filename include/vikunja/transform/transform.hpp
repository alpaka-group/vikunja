#pragma once

#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>
#include <vikunja/workdiv/BlockBasedWorkDiv.hpp>
#include <vikunja/transform/detail/BlockThreadTransformKernel.hpp>
#include <alpaka/alpaka.hpp>
#include <type_traits>

namespace vikunja
{
    namespace transform
    {
        /**
         * This is a function that transforms every element of an input iterator to another element in an output
         * iterator, i.e. if one has the array [1,2,3,4] and the transform function (x) -> x + 1, the output
         * will contain [2,3,4,5].
         * Input and output iterator can be the same. The output must be at least as big as the input, otherwise bad
         * things are bound to happen.
         * @tparam TAcc The alpaka accelerator type.
         * @tparam WorkDivPolicy The working division policy. Defaults to a templated value depending on the
         * accelerator. For the API of this, see workdiv/BlockBasedWorkDiv.hpp
         * @tparam MemAccessPolicy The memory access policy. Defaults to a templated value depending on the
         * accelerator. For the API of this, see mem/iterator/PolicyBasedBlockIterator
         * @tparam TFunc Type of the transform operator.
         * @tparam TInputIterator  Type of the input iterator. Should be a pointer-like type.
         * @tparam TOutputIterator Type of the output iterator. Should be a pointer-like type.
         * @tparam TDevAcc The type of the alpaka accelerator.
         * @tparam TQueue The type of the alpaka queue.
         * @tparam TIdx The index type to use.
         * @param devAcc The alpaka accelerator.
         * @param queue The alpaka queue.
         * @param n The number of input elements. Must be of type TIdx.
         * @param source The input iterator. Should be pointer-like.
         * @param destination The output iterator. Should be pointer-like.
         * @param func The transform operator.
         */
        template<
            typename TAcc,
            typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>,
            typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>,
            typename TFunc,
            typename TInputIterator,
            typename TOutputIterator,
            typename TDevAcc,
            typename TQueue,
            typename TIdx>
        auto deviceTransform(
            TDevAcc& devAcc,
            TQueue& queue,
            TIdx const& n,
            TInputIterator const& source,
            TOutputIterator const& destination,
            TFunc const& func) -> void
        {
            if(n == 0)
            {
                return;
            }
            constexpr uint64_t blockSize = WorkDivPolicy::template getBlockSize<TAcc>();
            using Dim = alpaka::Dim<TAcc>;
            using WorkDiv = alpaka::WorkDivMembers<Dim, TIdx>;
            using Vec = alpaka::Vec<Dim, TIdx>;
            constexpr TIdx xIndex = Dim::value - 1u;
            if(n < blockSize)
            {
                // TODO fix this?
                // maybe not needed
            }
            TIdx gridSize = WorkDivPolicy::template getGridSize<TAcc>(devAcc);
            TIdx maxGridSize = static_cast<TIdx>((((n + 1) / 2) - 1) / static_cast<TIdx>(blockSize) + 1);
            if(gridSize > maxGridSize)
            {
                gridSize = maxGridSize;
            }
            TIdx workDivGridSize = gridSize;
            TIdx workDivBlockSize = blockSize;

            Vec elementsPerThread(Vec::all(static_cast<TIdx>(1u)));
            Vec threadsPerBlock(Vec::all(static_cast<TIdx>(1u)));
            Vec blocksPerGrid(Vec::all(static_cast<TIdx>(1u)));

            blocksPerGrid[xIndex] = workDivGridSize;
            threadsPerBlock[xIndex] = workDivBlockSize;

            WorkDiv multiBlockWorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
            detail::BlockThreadTransformKernel<blockSize, MemAccessPolicy> kernel;
            alpaka::exec<TAcc>(queue, multiBlockWorkDiv, kernel, source, destination, n, func);
        }

        /**
         * A transform similar to the above, except that two input iterators are used in parallel.
         * @tparam TAcc
         * @tparam WorkDivPolicy
         * @tparam MemAccessPolicy
         * @tparam TFunc
         * @tparam TInputIterator
         * @tparam TInputIteratorSecond
         * @tparam TOutputIterator
         * @tparam TDevAcc
         * @tparam TQueue
         * @tparam TIdx
         * @param devAcc
         * @param queue
         * @param n
         * @param source
         * @param sourceSecond
         * @param destination
         * @param func
         */
        template<
            typename TAcc,
            typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>,
            typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>,
            typename TFunc,
            typename TInputIterator,
            typename TInputIteratorSecond,
            typename TOutputIterator,
            typename TDevAcc,
            typename TQueue,
            typename TIdx>
        auto deviceTransform(
            TDevAcc& devAcc,
            TQueue& queue,
            TIdx const& n,
            TInputIterator const& source,
            TInputIteratorSecond const& sourceSecond,
            TOutputIterator const& destination,
            TFunc const& func) -> void
        {
            if(n == 0)
            {
                return;
            }
            constexpr uint64_t blockSize = WorkDivPolicy::template getBlockSize<TAcc>();
            using Dim = alpaka::Dim<TAcc>;
            using WorkDiv = alpaka::WorkDivMembers<Dim, TIdx>;
            using Vec = alpaka::Vec<Dim, TIdx>;
            constexpr TIdx xIndex = Dim::value - 1u;
            if(n < blockSize)
            {
                // TODO fix this?
                // maybe not needed
            }
            TIdx gridSize = WorkDivPolicy::template getGridSize<TAcc>(devAcc);
            TIdx maxGridSize = static_cast<TIdx>((((n + 1) / 2) - 1) / static_cast<TIdx>(blockSize) + 1);
            if(gridSize > maxGridSize)
            {
                gridSize = maxGridSize;
            }
            TIdx workDivGridSize = gridSize;
            TIdx workDivBlockSize = blockSize;

            Vec elementsPerThread(Vec::all(static_cast<TIdx>(1u)));
            Vec threadsPerBlock(Vec::all(static_cast<TIdx>(1u)));
            Vec blocksPerGrid(Vec::all(static_cast<TIdx>(1u)));

            blocksPerGrid[xIndex] = workDivGridSize;
            threadsPerBlock[xIndex] = workDivBlockSize;

            WorkDiv multiBlockWorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
            detail::BlockThreadTransformKernel<blockSize, MemAccessPolicy> kernel;
            alpaka::exec<TAcc>(queue, multiBlockWorkDiv, kernel, source, sourceSecond, destination, n, func);
        }
    } // namespace transform
} // namespace vikunja
