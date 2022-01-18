/* Copyright 2021 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>
#include <vikunja/operators/operators.hpp>
#include <vikunja/reduce/detail/BlockThreadReduceKernel.hpp>
#include <vikunja/reduce/detail/SmallProblemReduceKernel.hpp>
#include <vikunja/workdiv/BlockBasedWorkDiv.hpp>

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <type_traits>

namespace vikunja
{
    namespace reduce
    {
        namespace detail
        {
            /**
             * This provides an identity function that returns whatever argument it is given. It is used for the
             * deviceReduce.
             * @tparam T Any type.
             */
            template<typename T>
            struct Identity
            {
                /**
                 * The identity function.
                 * @param arg Any argument.
                 * @return The parameter arg.
                 */
                constexpr ALPAKA_FN_HOST_ACC T operator()(T const& arg) const
                {
                    return arg;
                }
            };
        } // namespace detail

        /**
         * This is a function which transforms the input values and uses a reduce to accumulate the transformed values.
         * For example, given the array [1, 2, 3, 4], the transform function (x) -> x + 1, and the reduce function
         * (x,y) -> x + y would return 2 + 3 + 4 + 5 = 14.
         * @tparam TAcc The alpaka accelerator type to use.
         * @tparam WorkDivPolicy The working division policy. Defaults to a templated value depending on the
         * accelerator. For the API of this, see workdiv/BlockBasedWorkDiv.hpp
         * @tparam MemAccessPolicy The memory access policy. Defaults to a templated value depending on the
         * accelerator. For the API of this, see mem/iterator/PolicyBasedBlockIterator
         * @tparam TTransformFunc Type of the transform operator.
         * @tparam TReduceFunc Type of the reduce operator.
         * @tparam TInputIterator Type of the input iterator. Should be a pointer-like type.
         * @tparam TDevAcc The type of the alpaka accelerator.
         * @tparam TDevHost The type of the alpaka host.
         * @tparam TQueue The type of the alpaka queue.
         * @tparam TIdx The index type to use.
         * @tparam TTransformOperator The vikunja::operators type of the transform function.
         * @tparam TReduceOperator The vikunja::operators type of the reduce function.
         * @tparam TRed The return value of the function.
         * @param devAcc The alpaka accelerator.
         * @param devHost The alpaka host.
         * @param queue The alpaka queue.
         * @param n The number of input elements. Must be of type TIdx.
         * @param buffer The input iterator. Should be a pointer-like object.
         * @param transformFunc The transform operator.
         * @param reduceFunc The reduce operator.
         * @return Value of the combined transform/reduce operation.
         */
        template<
            typename TAcc,
            typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>,
            typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>,
            typename TTransformFunc,
            typename TReduceFunc,
            typename TInputIterator,
            typename TDevAcc,
            typename TDevHost,
            typename TQueue,
            typename TIdx,
            typename TTransformOperator = vikunja::operators::
                UnaryOp<TAcc, TTransformFunc, typename std::iterator_traits<TInputIterator>::value_type>,
            typename TReduceOperator = vikunja::operators::
                BinaryOp<TAcc, TReduceFunc, typename TTransformOperator::TRed, typename TTransformOperator::TRed>,
            typename TRed = typename TReduceOperator::TRed>
        auto deviceTransformReduce(
            TDevAcc& devAcc,
            TDevHost& devHost,
            TQueue& queue,
            TIdx const& n,
            TInputIterator const& buffer,
            TTransformFunc const& transformFunc,
            TReduceFunc const& reduceFunc) -> TRed
        {
            // ok, now we have to think about what to do now
            // TODO: think of proper solution for this.
            // TODO: This actually needs discussion: As no default value is provided, the result is undefined.
            if(n == 0)
            {
                //            return static_cast<TRed>(0);
            }
            constexpr uint64_t blockSize = WorkDivPolicy::template getBlockSize<TAcc>();
            using Dim = alpaka::Dim<TAcc>;
            using WorkDiv = alpaka::WorkDivMembers<Dim, TIdx>;
            using Vec = alpaka::Vec<Dim, TIdx>;
            constexpr TIdx xIndex = Dim::value - 1u;

            Vec elementsPerThread(Vec::all(static_cast<TIdx>(1u)));
            Vec threadsPerBlock(Vec::all(static_cast<TIdx>(1u)));
            Vec blocksPerGrid(Vec::all(static_cast<TIdx>(1u)));

            Vec const resultBufferExtent(Vec::all(static_cast<TIdx>(1u)));
            auto resultBuffer(alpaka::allocBuf<TRed, TIdx>(devAcc, resultBufferExtent));

            // in case n < blockSize, the block reductions only work
            // if the MemAccessPolicy maps the correct values.
            if(n < blockSize)
            {
                auto resultBuffer(alpaka::allocBuf<TRed, TIdx>(devAcc, resultBufferExtent));
                WorkDiv dummyWorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
                detail::SmallProblemReduceKernel<TTransformOperator, TReduceOperator> kernel;
                alpaka::exec<TAcc>(
                    queue,
                    dummyWorkDiv,
                    kernel,
                    buffer,
                    alpaka::getPtrNative(resultBuffer),
                    n,
                    transformFunc,
                    reduceFunc);
                auto resultView(alpaka::allocBuf<TRed, TIdx>(devHost, resultBufferExtent));
                // TRed result;
                // alpaka::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost,
                // static_cast<TIdx>(1u)};
                alpaka::memcpy(queue, resultView, resultBuffer, resultBufferExtent);
                alpaka::wait(queue);
                auto result = alpaka::getPtrNative(resultView);
                return result[0];
            }


            TIdx gridSize = WorkDivPolicy::template getGridSize<TAcc>(devAcc);

            TIdx maxGridSize = static_cast<TIdx>((((n + 1) / 2) - 1) / static_cast<TIdx>(blockSize) + 1);
            if(gridSize > maxGridSize)
            {
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

            WorkDiv multiBlockWorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
            WorkDiv singleBlockWorkDiv{singleBlocksPerGrid, singleThreadsPerBlock, singleElementsPerThread};

            auto secondPhaseBuffer(alpaka::allocBuf<TRed, TIdx>(devAcc, sharedMemExtent));

            detail::BlockThreadReduceKernel<blockSize, MemAccessPolicy, TRed, TTransformOperator, TReduceOperator>
                multiBlockKernel;

            using TIdentityTransformOperator
                = vikunja::operators::UnaryOp<TAcc, detail::Identity<TRed>, typename TTransformOperator::TRed>;
            detail::
                BlockThreadReduceKernel<blockSize, MemAccessPolicy, TRed, TIdentityTransformOperator, TReduceOperator>
                    singleBlockKernel;
            // execute kernels
            alpaka::exec<TAcc>(
                queue,
                multiBlockWorkDiv,
                multiBlockKernel,
                buffer,
                alpaka::getPtrNative(secondPhaseBuffer),
                n,
                transformFunc,
                reduceFunc);
            alpaka::exec<TAcc>(
                queue,
                singleBlockWorkDiv,
                singleBlockKernel,
                alpaka::getPtrNative(secondPhaseBuffer),
                alpaka::getPtrNative(secondPhaseBuffer),
                gridSize,
                detail::Identity<TRed>(),
                reduceFunc);

            auto resultView(alpaka::allocBuf<TRed, TIdx>(devHost, resultBufferExtent));
            alpaka::memcpy(queue, resultView, secondPhaseBuffer, resultBufferExtent);

            // wait for result, otherwise the async CPU queue causes a segfault
            alpaka::wait(queue);

            auto result = alpaka::getPtrNative(resultView);
            return result[0];
        }

        /**
         * This is a function which transforms the input values and uses a reduce to accumulate the transformed values.
         * For example, given the array [1, 2, 3, 4], the transform function (x) -> x + 1, and the reduce function
         * (x,y) -> x + y would return 2 + 3 + 4 + 5 = 14.
         * @tparam TAcc The alpaka accelerator type to use.
         * @tparam WorkDivPolicy The working division policy. Defaults to a templated value depending on the
         * accelerator. For the API of this, see workdiv/BlockBasedWorkDiv.hpp
         * @tparam MemAccessPolicy The memory access policy. Defaults to a templated value depending on the
         * accelerator. For the API of this, see mem/iterator/PolicyBasedBlockIterator
         * @tparam TTransformFunc Type of the transform operator.
         * @tparam TReduceFunc Type of the reduce operator.
         * @tparam TInputIterator Type of the input iterator. Should be a pointer-like type.
         * @tparam TDevAcc The type of the alpaka accelerator.
         * @tparam TDevHost The type of the alpaka host.
         * @tparam TQueue The type of the alpaka queue.
         * @tparam TTransformOperator The vikunja::operators type of the transform function.
         * @tparam TReduceOperator The vikunja::operators type of the reduce function.
         * @tparam TRed The return value of the function.
         * @param devAcc The alpaka accelerator.
         * @param devHost The alpaka host.
         * @param queue The alpaka queue.
         * @param bufferBegin The begin pointer of the input buffer.
         * @param bufferEnd The end pointer of the input buffer.
         * @param transformFunc The transform operator.
         * @param reduceFunc The reduce operator.
         * @return Value of the combined transform/reduce operation.
         */
        template<
            typename TAcc,
            typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>,
            typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>,
            typename TTransformFunc,
            typename TReduceFunc,
            typename TInputIterator,
            typename TDevAcc,
            typename TDevHost,
            typename TQueue,
            typename TTransformOperator = vikunja::operators::
                UnaryOp<TAcc, TTransformFunc, typename std::iterator_traits<TInputIterator>::value_type>,
            typename TReduceOperator = vikunja::operators::
                BinaryOp<TAcc, TReduceFunc, typename TTransformOperator::TRed, typename TTransformOperator::TRed>,
            typename TRed = typename TReduceOperator::TRed>
        auto deviceTransformReduce(
            TDevAcc& devAcc,
            TDevHost& devHost,
            TQueue& queue,
            TInputIterator const& bufferBegin,
            TInputIterator const& bufferEnd,
            TTransformFunc const& transformFunc,
            TReduceFunc const& reduceFunc) -> TRed
        {
            assert(bufferEnd >= bufferBegin);
            auto size = static_cast<typename alpaka::traits::IdxType<TAcc>::type>(bufferEnd - bufferBegin);
            return deviceTransformReduce<TAcc>(devAcc, devHost, queue, size, bufferBegin, transformFunc, reduceFunc);
        }

        /**
         * This is a reduce function, which works exactly like deviceTransformReduce with an identity function for
         * the transform operator.
         * @see deviceTransformReduce.
         * @tparam TAcc
         * @tparam WorkDivPolicy
         * @tparam MemAccessPolicy
         * @tparam TFunc
         * @tparam TInputIterator
         * @tparam TDevAcc
         * @tparam TDevHost
         * @tparam TQueue
         * @tparam TIdx
         * @tparam TReduceOperator The vikunja::operators type of the reduce function.
         * @tparam TRed The return value of the function.
         * @param devAcc
         * @param devHost
         * @param queue
         * @param n
         * @param buffer
         * @param func
         * @return
         */
        template<
            typename TAcc,
            typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>,
            typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>,
            typename TFunc,
            typename TInputIterator,
            typename TDevAcc,
            typename TDevHost,
            typename TQueue,
            typename TIdx,
            typename TOperator = vikunja::operators::BinaryOp<
                TAcc,
                TFunc,
                typename std::iterator_traits<TInputIterator>::value_type,
                typename std::iterator_traits<TInputIterator>::value_type>,
            typename TRed = typename TOperator::TRed>
        auto deviceReduce(
            TDevAcc& devAcc,
            TDevHost& devHost,
            TQueue& queue,
            TIdx const& n,
            TInputIterator const& buffer,
            TFunc const& func) -> TRed
        {
            return deviceTransformReduce<
                TAcc,
                WorkDivPolicy,
                MemAccessPolicy,
                detail::Identity<TRed>,
                TFunc,
                TInputIterator,
                TDevAcc,
                TDevHost,
                TQueue,
                TIdx>(devAcc, devHost, queue, n, buffer, detail::Identity<TRed>(), func);
        }

        /**
         * This is a reduce function, which works exactly like deviceTransformReduce with an identity function for
         * the transform operator.
         * @see deviceTransformReduce.
         * @tparam TAcc
         * @tparam WorkDivPolicy
         * @tparam MemAccessPolicy
         * @tparam TFunc
         * @tparam TInputIterator
         * @tparam TDevAcc
         * @tparam TDevHost
         * @tparam TQueue
         * @tparam TReduceOperator The vikunja::operators type of the reduce function.
         * @tparam TRed The return value of the function.
         * @param devAcc
         * @param devHost
         * @param queue
         * @param bufferBegin The begin pointer of the input buffer.
         * @param bufferEnd The end pointer of the input buffer.
         * @param func
         * @return
         */
        template<
            typename TAcc,
            typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>,
            typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>,
            typename TFunc,
            typename TInputIterator,
            typename TDevAcc,
            typename TDevHost,
            typename TQueue,
            typename TOperator = vikunja::operators::BinaryOp<
                TAcc,
                TFunc,
                typename std::iterator_traits<TInputIterator>::value_type,
                typename std::iterator_traits<TInputIterator>::value_type>,
            typename TRed = typename TOperator::TRed>
        auto deviceReduce(
            TDevAcc& devAcc,
            TDevHost& devHost,
            TQueue& queue,
            TInputIterator const& bufferBegin,
            TInputIterator const& bufferEnd,
            TFunc const& func) -> TRed
        {
            assert(bufferEnd >= bufferBegin);
            auto size = static_cast<typename alpaka::traits::IdxType<TAcc>::type>(bufferEnd - bufferBegin);
            return deviceReduce<TAcc>(devAcc, devHost, queue, size, bufferBegin, func);
        }
    } // namespace reduce
} // namespace vikunja
