/* Copyright 2021 Hauke Mewes
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

namespace vikunja
{
    namespace reduce
    {
        namespace detail
        {
            /**
             * This is a sequential reduce kernel that is used for small problem sizes.
             * @tparam TTransformOperator The vikunja::operators type of the transform function.
             * @tparam TReduceOperator The vikunja::operators type of the reduce function.
             */
            template<typename TTransformOperator, typename TReduceOperator>
            struct SmallProblemReduceKernel
            {
                template<
                    typename TAcc,
                    typename TIdx,
                    typename TInputIterator,
                    typename TOutputIterator,
                    typename TTransformFunc,
                    typename TReduceFunc>
                ALPAKA_FN_ACC void operator()(
                    TAcc const& acc __attribute__((unused)),
                    TInputIterator const& source,
                    TOutputIterator const& destination,
                    TIdx const& n,
                    TTransformFunc const& transformFunc,
                    TReduceFunc const& reduceFunc) const
                {
                    auto tSum = TTransformOperator::run(acc, transformFunc, *(source));
                    for(TIdx i(1); i < n; ++i)
                    {
                        tSum = TReduceOperator::run(
                            acc,
                            reduceFunc,
                            tSum,
                            TTransformOperator::run(acc, transformFunc, *(source + i)));
                    }
                    *destination = tSum;
                }
            };
        } // namespace detail
    } // namespace reduce
} // namespace vikunja
