//
// Created by mewes30 on 19.12.18.
//

#pragma once

#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>
#include <alpaka/alpaka.hpp>

namespace vikunja {
namespace reduce {
namespace detail {

    template<typename TRed, uint64_t size>
    struct sharedStaticArray
    {
        TRed data[size];

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE TRed &operator[](uint64_t index) {
            return data[index];
        }
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const TRed &operator[](uint64_t index) const {
            return data[index];
        }
    };

    template<uint64_t TBlockSize, typename TMemAccessPolicy, typename TRed, typename TFunc>
    struct BlockThreadReduceKernel {
        template<typename TAcc, typename TIdx,
                typename TInputIterator, typename TOutputIterator>
        ALPAKA_FN_ACC void operator()(TAcc const &acc,
                TInputIterator const * const source,
                TOutputIterator destination,
                TIdx const &n,
                TFunc func) const  {
            // Shared Mem
            auto &sdata(
                    alpaka::block::shared::st::allocVar<sharedStaticArray<TRed, TBlockSize>,
                    __COUNTER__>(acc));


            // blockIdx.x
            auto blockIndex = (alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]);
            // threadIdx.x
            auto threadIndex = (alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]);
            // gridDim.x
            auto gridDimension = (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0]);

            // blockIdx.x * TBlocksize + threadIdx.x
            auto indexInBlock(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]);

            using MemPolicy = TMemAccessPolicy;
            vikunja::mem::iterator::PolicyBasedBlockIterator<MemPolicy, TAcc, TInputIterator> iter(source, acc, n, TBlockSize);
            //std::cout << "threadIndex: " << threadIndex << "\n";
            // sequential
            auto startIndex = MemPolicy::getStartIndex(acc, n, TBlockSize);
            auto endIndex = MemPolicy::getEndIndex(acc, n, TBlockSize);
            auto stepSize = MemPolicy::getStepSize(acc, n, TBlockSize);

            // WARNING: in theory, one might return here, but then the cpu kernels get stuck on the syncthreads.
            /*if(iter >= iter.end()) {
               // std::cout << "In return: startIndex: " + std::to_string(startIndex) + ", endIndex: " + std::to_string(endIndex) + ", threadIndex: " + std::to_string(threadIndex) +"\n";
               // return;
            } else {
               // std::cout << "startIndex: " + std::to_string(startIndex) + ", endIndex: " + std::to_string(endIndex) + "\n";
            }*/
           // auto start = std::chrono::high_resolution_clock::now();
            auto tSum = *iter;
            ++iter;
            while(iter + 3 < iter.end()) {
                tSum = func(func(func(func(tSum, *iter), *(iter + 1)), *(iter + 2)), *(iter + 3));
                iter += 4;
            }
            while(iter < iter.end()) {
                tSum = func(tSum, *iter);
                ++iter;
            }
            //auto endTime = std::chrono::high_resolution_clock::now();
            //std::cout << std::chrono::duration_cast<std::chrono::microseconds>(endTime - start).count() << " microseconds\n";
/*
            auto i = startIndex;
            if(i >= n) {
                return;
            }

            auto tSum = *(source + i);
            i += stepSize;
            // Level 1: Grid reduce, reading from global memory
            while(i < endIndex) {
                tSum = func(tSum, *(source + i));
                i += stepSize;
            }*/
            if(threadIndex < n) {
                sdata[threadIndex] = tSum;
            }

            alpaka::block::sync::syncBlockThreads(acc);
           // std::cout << "Got to here\n";
            // blockReduce
            // unroll for better performance
            for(TIdx bs = TBlockSize, bSup = (TBlockSize + 1) / 2;
            bs > 1; bs = bs / 2, bSup = (bs + 1) / 2) {
                //std::cout << ("Amokthread: " + std::to_string(threadIndex) + "\n");
                bool condition = threadIndex < bSup && // only first half of block is working
                         (threadIndex + bSup) < TBlockSize && // index for second half must be in bounds
                         (indexInBlock + bSup) < n; // if element in second half has ben initialized before
                if(condition) {
                    sdata[threadIndex] = func(sdata[threadIndex], sdata[threadIndex + bSup]);
                }
                //std::cout << "Before second block\n";
                alpaka::block::sync::syncBlockThreads(acc);
                //std::cout << "After second block\n";
            }
            //std::cout << "After loop\n";
            if(threadIndex == 0) {
                *(destination + blockIndex) = sdata[0];
            }
            //std::cout << "Triggered last statement\n";
        }
    };
}
}
}