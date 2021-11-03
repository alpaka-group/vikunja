/* Copyright 2021 Hauke Mewes
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>

#include <alpaka/alpaka.hpp>

namespace vikunja
{
    namespace transform
    {
        namespace detail
        {
            /**
             * This provides transform kernels for both the single-input and the double-input transform operation.
             * @tparam TBlockSize The block size of the kernel.
             * @tparam TMemAccessPolicy The memory access policy of the kernel.
             * @tparam TOperator The vikunja::operators type of the transform function.
             */
            template<uint64_t TBlockSize, typename TMemAccessPolicy, typename TOperator>
            struct BlockThreadTransformKernel
            {
                template<
                    typename TAcc,
                    typename TIdx,
                    typename TInputIterator,
                    typename TOutputIterator,
                    typename TFunc>
                ALPAKA_FN_ACC void operator()(
                    TAcc const& acc,
                    TInputIterator const& source,
                    TOutputIterator const& destination,
                    TIdx const& n,
                    TFunc const& func) const
                {
                    vikunja::mem::iterator::PolicyBasedBlockIterator<TMemAccessPolicy, TAcc, TInputIterator>
                        inputIterator(source, acc, n, TBlockSize);
                    vikunja::mem::iterator::PolicyBasedBlockIterator<TMemAccessPolicy, TAcc, TOutputIterator>
                        outputIterator(destination, acc, n, TBlockSize);

                    while(inputIterator < inputIterator.end())
                    {
                        *outputIterator = TOperator::run(acc, func, *inputIterator);
                        ++inputIterator;
                        ++outputIterator;
                    }
                }

                template<
                    typename TAcc,
                    typename TIdx,
                    typename TInputIterator,
                    typename TInputIteratorSecond,
                    typename TOutputIterator,
                    typename TFunc>
                ALPAKA_FN_ACC void operator()(
                    TAcc const& acc,
                    TInputIterator const& source,
                    TInputIteratorSecond const& sourceSecond,
                    TOutputIterator const& destination,
                    TIdx const& n,
                    TFunc const& func) const
                {
                    vikunja::mem::iterator::PolicyBasedBlockIterator<TMemAccessPolicy, TAcc, TInputIterator>
                        inputIterator(source, acc, n, TBlockSize);
                    vikunja::mem::iterator::PolicyBasedBlockIterator<TMemAccessPolicy, TAcc, TInputIteratorSecond>
                        inputIteratorSecond(sourceSecond, acc, n, TBlockSize);
                    vikunja::mem::iterator::PolicyBasedBlockIterator<TMemAccessPolicy, TAcc, TOutputIterator>
                        outputIterator(destination, acc, n, TBlockSize);

                    while(inputIterator < inputIterator.end())
                    {
                        *outputIterator = TOperator::run(acc, func, *inputIterator, *inputIteratorSecond);
                        ++inputIterator;
                        ++inputIteratorSecond;
                        ++outputIterator;
                    }
                }
            };
        } // namespace detail
    } // namespace transform
} // namespace vikunja
