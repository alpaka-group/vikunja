#pragma once

#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>
#include <alpaka/alpaka.hpp>

namespace vikunja
{
    namespace reduce
    {
        namespace detail
        {
            /**
             * A helper static array for the shared memory. This wrapper is necessary as alpaka does not allow arrays
             * as shared memory directly.
             * @tparam TRed The data type of the array.
             * @tparam size The array size.
             */
            template<typename TRed, uint64_t size>
            struct sharedStaticArray
            {
                TRed data[size];

                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE TRed& operator[](uint64_t index)
                {
                    return data[index];
                }
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const TRed& operator[](uint64_t index) const
                {
                    return data[index];
                }
            };


            /**
             * This is the block reduce kernel operator class.
             * @tparam TBlockSize The block size of this reduce kernel.
             * @tparam TMemAccessPolicy The memory access policy of this reduce kernel.
             * @tparam TRed The type of the reduction.
             */
            template<uint64_t TBlockSize, typename TMemAccessPolicy, typename TRed>
            struct BlockThreadReduceKernel
            {
                /**
                 * This is the block reduce kernel operator.
                 * @tparam TAcc The alpaka accelerator type.
                 * @tparam TIdx The type of the access index.
                 * @tparam TInputIterator The input iterator type, should be pointer-like.
                 * @tparam TOutputIterator The helper memory output iterator type, should be pointer-like.
                 * @tparam TTransformFunc The transform operator type.
                 * @tparam TFunc The reduce operator type.
                 * @param acc The alpaka accelerator.
                 * @param source The input iterator.
                 * @param destination The helper memory output iterator.
                 * @param n The size of the input iterator.
                 * @param transformFunc The transform operator.
                 * @param func THe reduce operator.
                 */
                template<
                    typename TAcc,
                    typename TIdx,
                    typename TInputIterator,
                    typename TOutputIterator,
                    typename TTransformFunc,
                    typename TFunc>
                ALPAKA_FN_ACC void operator()(
                    TAcc const& acc,
                    TInputIterator const& source,
                    TOutputIterator const& destination,
                    TIdx const& n,
                    TTransformFunc const& transformFunc,
                    TFunc const& func) const
                {
                    // use shared memory in this block for the reduce.
                    auto& sdata(alpaka::declareSharedVar<sharedStaticArray<TRed, TBlockSize>, __COUNTER__>(acc));

                    // alpaka reverses the order of the cuda x/y/z parametors:
                    // If 3d acc is used, 0 is equivalent to z, 1 to y, 2 to x.
                    constexpr TIdx xIndex = alpaka::Dim<TAcc>::value - 1u;

                    // CUDA equivalents:
                    // blockIdx.x
                    auto blockIndex = (alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[xIndex]);
                    // threadIdx.x
                    auto threadIndex = (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[xIndex]);
                    // blockIdx.x * TBlocksize + threadIdx.x
                    auto indexInBlock(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[xIndex]);

                    using MemPolicy = TMemAccessPolicy;
                    // Create an iterator with the specified memory access policy that wraps the input iterator.
                    vikunja::mem::iterator::PolicyBasedBlockIterator<MemPolicy, TAcc, TInputIterator> iter(
                        source,
                        acc,
                        n,
                        TBlockSize);
                    auto startIndex
                        = MemPolicy::getStartIndex(acc, static_cast<TIdx>(n), static_cast<TIdx>(TBlockSize));
                    // only do work if the index is in bounds.
                    // One might want to move that to a property of the iterator, like iter.isValid or something like
                    // this.
                    if(startIndex < n)
                    {
                        // no neutral element is used, so initialize with value from first element.
                        auto tSum = transformFunc(acc, *iter);
                        ++iter;
                        // Manual unrolling. I dont know if this is really necessary, but
                        while(iter + 3 < iter.end())
                        {
                            tSum = func(
                                acc,
                                func(
                                    acc,
                                    func(
                                        acc,
                                        func(acc, tSum, transformFunc(acc, *iter)),
                                        transformFunc(acc, *(iter + 1))),
                                    transformFunc(acc, *(iter + 2))),
                                transformFunc(acc, *(iter + 3)));
                            iter += 4;
                        }
                        while(iter < iter.end())
                        {
                            tSum = func(acc, tSum, transformFunc(acc, *iter));
                            ++iter;
                        }
                        // This condition actually relies on the memory access pattern.
                        // When gridStriding is used, the first n threads always get the first n values,
                        // but when the linearMemAccess is used, they do not.
                        // This is circumvented by now that if the block size is bigger than the problem size, a
                        // sequential algorithm is used instead.
                        if(MemPolicy::isValidThreadResult(acc, static_cast<TIdx>(n), static_cast<TIdx>(n)))
                        {
                            sdata[threadIndex] = tSum;
                        }
                    }

                    alpaka::syncBlockThreads(acc);

                    // blockReduce
                    // unroll for better performance
                    for(TIdx bs = TBlockSize, bSup = (TBlockSize + 1) / 2; bs > 1; bs = bs / 2, bSup = (bs + 1) / 2)
                    {
                        bool condition = threadIndex < bSup && // only first half of block is working
                            (threadIndex + bSup) < TBlockSize && // index for second half must be in bounds
                            (indexInBlock + bSup) < n; // if element in second half has ben initialized before
                        if(condition)
                        {
                            sdata[threadIndex] = func(acc, sdata[threadIndex], sdata[threadIndex + bSup]);
                        }
                        alpaka::syncBlockThreads(acc); // sync: block reduce loop
                    }
                    if(threadIndex == 0)
                    {
                        *(destination + blockIndex) = sdata[0];
                    }
                }
            };
        } // namespace detail
    } // namespace reduce
} // namespace vikunja
