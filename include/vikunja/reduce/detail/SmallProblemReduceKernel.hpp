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
             */
            struct SmallProblemReduceKernel
            {
                template<
                    typename TAcc,
                    typename TIdx,
                    typename TInputIterator,
                    typename TOutputIterator,
                    typename TTransformFunc,
                    typename TFunc>
                ALPAKA_FN_ACC void operator()(
                    TAcc const& acc __attribute__((unused)),
                    TInputIterator const& source,
                    TOutputIterator const& destination,
                    TIdx const& n,
                    TTransformFunc const& transformFunc,
                    TFunc const& func) const
                {
                    auto tSum = transformFunc(acc, *(source));
                    for(TIdx i(1); i < n; ++i)
                    {
                        tSum = func(acc, tSum, transformFunc(acc, *(source + i)));
                    }
                    *destination = tSum;
                }
            };
        } // namespace detail
    } // namespace reduce
} // namespace vikunja
