/* Copyright 2022 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <vikunja/access/BlockStrategy.hpp>
#include <vikunja/concept/operator.hpp>
#include <vikunja/transform/detail/BlockThreadTransformKernel.hpp>
#include <vikunja/workdiv/BlockBasedWorkDiv.hpp>

#include <alpaka/alpaka.hpp>

#include <type_traits>

// @ALPAKA_BACKWARD(<=0.8)
// in alpaka 0.9, the namespace traits was renamed to trait
// https://github.com/alpaka-group/alpaka/pull/1651
// enable backwards compatibility
#if ALPAKA_VERSION_MAJOR == 0 && ALPAKA_VERSION_MINOR < 9
namespace alpaka
{
    namespace trait = ::alpaka::traits;
}
#endif

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
         * accelerator. For the API of this, see vikunja::MemAccess::PolicyBasedBlockStrategy
         * @tparam TFunc Type of the transform operator.
         * @tparam TInputIterator  Type of the input iterator. Should be a pointer-like type.
         * @tparam TOutputIterator Type of the output iterator. Should be a pointer-like type.
         * @tparam TDevAcc The type of the alpaka accelerator.
         * @tparam TQueue The type of the alpaka queue.
         * @tparam TIdx The index type to use.
         * @tparam TOperator The specialization of vikunja::concept::UnaryOp type of the transform function.
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
            typename MemAccessPolicy = vikunja::MemAccess::MemAccessPolicy<TAcc>,
            typename TFunc,
            typename TInputIterator,
            typename TOutputIterator,
            typename TDevAcc,
            typename TQueue,
            typename TIdx,
            typename TOperator
            = vikunja::concept::UnaryOp<TAcc, TFunc, typename std::iterator_traits<TInputIterator>::value_type>>
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
            detail::BlockThreadTransformKernel<blockSize, MemAccessPolicy, TOperator> kernel;
            alpaka::exec<TAcc>(queue, multiBlockWorkDiv, kernel, source, destination, n, func);
        }

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
         * accelerator. For the API of this, see vikunja::MemAccess::PolicyBasedBlockStrategy
         * @tparam TFunc Type of the transform operator.
         * @tparam TInputIterator  Type of the input iterator. Should be a pointer-like type.
         * @tparam TOutputIterator Type of the output iterator. Should be a pointer-like type.
         * @tparam TDevAcc The type of the alpaka accelerator.
         * @tparam TQueue The type of the alpaka queue.
         * @tparam TOperator The specialization of vikunja::concept::UnaryOp type of the transform function.
         * @param devAcc The alpaka accelerator.
         * @param queue The alpaka queue.
         * @param sourceBegin The begin pointer of the input buffer.
         * @param sourceEnd The end pointer of the input buffer.
         * @param destination The output iterator. Should be pointer-like.
         * @param func The transform operator.
         */
        template<
            typename TAcc,
            typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>,
            typename MemAccessPolicy = vikunja::MemAccess::MemAccessPolicy<TAcc>,
            typename TFunc,
            typename TInputIterator,
            typename TOutputIterator,
            typename TDevAcc,
            typename TQueue,
            typename TOperator
            = vikunja::concept::UnaryOp<TAcc, TFunc, typename std::iterator_traits<TInputIterator>::value_type>>
        auto deviceTransform(
            TDevAcc& devAcc,
            TQueue& queue,
            TInputIterator const& sourceBegin,
            TInputIterator const& sourceEnd,
            TOutputIterator const& destination,
            TFunc const& func) -> void
        {
            assert(sourceEnd >= sourceBegin);
            auto size = static_cast<typename alpaka::trait::IdxType<TAcc>::type>(sourceEnd - sourceBegin);
            deviceTransform<TAcc>(devAcc, queue, size, sourceBegin, destination, func);
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
         * @tparam TOperator
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
            typename MemAccessPolicy = vikunja::MemAccess::MemAccessPolicy<TAcc>,
            typename TFunc,
            typename TInputIterator,
            typename TInputIteratorSecond,
            typename TOutputIterator,
            typename TDevAcc,
            typename TQueue,
            typename TIdx,
            typename TOperator = vikunja::concept::BinaryOp<
                TAcc,
                TFunc,
                typename std::iterator_traits<TInputIterator>::value_type,
                typename std::iterator_traits<TInputIteratorSecond>::value_type>>
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
            detail::BlockThreadTransformKernel<blockSize, MemAccessPolicy, TOperator> kernel;
            alpaka::exec<TAcc>(queue, multiBlockWorkDiv, kernel, source, sourceSecond, destination, n, func);
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
         * @tparam TOperator
         * @param devAcc
         * @param queue
         * @param sourceBegin The begin pointer of the input buffer.
         * @param sourceEnd The end pointer of the input buffer.
         * @param sourceSecond
         * @param destination
         * @param func
         */
        template<
            typename TAcc,
            typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>,
            typename MemAccessPolicy = vikunja::MemAccess::MemAccessPolicy<TAcc>,
            typename TFunc,
            typename TInputIterator,
            typename TInputIteratorSecond,
            typename TOutputIterator,
            typename TDevAcc,
            typename TQueue,
            typename TOperator = vikunja::concept::BinaryOp<
                TAcc,
                TFunc,
                typename std::iterator_traits<TInputIterator>::value_type,
                typename std::iterator_traits<TInputIteratorSecond>::value_type>>
        auto deviceTransform(
            TDevAcc& devAcc,
            TQueue& queue,
            TInputIterator const& sourceBegin,
            TInputIterator const& sourceEnd,
            TInputIteratorSecond const& sourceSecond,
            TOutputIterator const& destination,
            TFunc const& func) -> void
        {
            assert(sourceEnd >= sourceBegin);
            auto size = static_cast<typename alpaka::trait::IdxType<TAcc>::type>(sourceEnd - sourceBegin);
            deviceTransform<TAcc>(devAcc, queue, size, sourceBegin, sourceSecond, destination, func);
        }
    } // namespace transform
} // namespace vikunja
