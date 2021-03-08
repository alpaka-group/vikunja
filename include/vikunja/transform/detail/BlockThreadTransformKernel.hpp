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
             */
            template<uint64_t TBlockSize, typename TMemAccessPolicy>
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
                        *outputIterator = func(acc, *inputIterator);
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
                        *outputIterator = func(acc, *inputIterator, *inputIteratorSecond);
                        ++inputIterator;
                        ++inputIteratorSecond;
                        ++outputIterator;
                    }
                }
            };
        } // namespace detail
    } // namespace transform
} // namespace vikunja
