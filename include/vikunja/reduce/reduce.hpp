#pragma once

#include <alpaka/alpaka.hpp>
#include <type_traits>
#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>
#include <vikunja/workdiv/BlockBasedWorkDiv.hpp>
#include <vikunja/reduce/detail/SmallProblemReduceKernel.hpp>
#include <vikunja/reduce/detail/BlockThreadReduceKernel.hpp>
#include <iostream>

namespace vikunja {
namespace reduce {
    namespace detail {
        /**
         * This provides an identity function that returns whatever argument it is given. It is used for the
         * deviceReduce.
         * @tparam T Any type.
         */
        template <typename T>
        struct Identity {
            /**
             * The identity function.
             * @param arg Any argument.
             * @return The parameter arg.
             */
            constexpr ALPAKA_FN_HOST_ACC T operator()(T const &arg) const {
                return arg;
            }
        };
    }

    /**
     * This is a function which transforms the input values and uses a reduce to accumulate the transformed values.
     * For example, given the array [1, 2, 3, 4], the transform function (x) -> x + 1, and the reduce function
     * (x,y) -> x + y would return 2 + 3 + 4 + 5 = 14.
     * @tparam TAcc The alpaka accelerator type to use.
     * @tparam WorkDivPolicy The working division policy. Defaults to a templated value depending on the accelerator.
     * For the API of this, see workdiv/BlockBasedWorkDiv.hpp
     * @tparam MemAccessPolicy The memory access policy. Defaults to a templated value depending on the accelerator.
     * For the API of this, see mem/iterator/PolicyBasedBlockIterator
     * @tparam TTransformFunc Type of the transform operator.
     * @tparam TFunc Type of the reduce operator.
     * @tparam TInputIterator Type of the input iterator. Should be a pointer-like type.
     * @tparam TDevAcc The type of the alpaka accelerator.
     * @tparam TDevHost The type of the alpaka host.
     * @tparam TQueue The type of the alpaka queue.
     * @tparam TIdx The index type to use.
     * @param devAcc The alpaka accelerator.
     * @param devHost The alpaka host.
     * @param queue The alpaka queue.
     * @param n The number of input elements. Must be of type TIdx.
     * @param buffer The input iterator. Should be a pointer-like object.
     * @param transformFunc The transform operator.
     * @param func The reduce operator.
     * @return Value of the combined transform/reduce operation.
     */
    template<typename TAcc, typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>, typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>, typename TTransformFunc, typename TFunc, typename TInputIterator, typename TDevAcc, typename TDevHost, typename TQueue, typename TIdx >
    auto deviceTransformReduce(TDevAcc &devAcc, TDevHost &devHost, TQueue &queue,  TIdx const &n, TInputIterator const &buffer, TTransformFunc const &transformFunc, TFunc const &func) -> decltype(func(transformFunc(*buffer), transformFunc(*buffer))) {

        // TODO: more elegant way to obtain return type + avoid that double declaration
        using TRed = decltype(func(transformFunc(*buffer), transformFunc(*buffer)));
        // ok, now we have to think about what to do now
        // TODO: think of proper solution for this.
        // TODO: This actually needs discussion: As no default value is provided, the result is undefined.
        if(n == 0) {
//            return static_cast<TRed>(0);
        }
        constexpr uint64_t blockSize = WorkDivPolicy::template getBlockSize<TAcc>();
        using Dim = alpaka::dim::Dim<TAcc>;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, TIdx>;
        using Vec = alpaka::vec::Vec<Dim, TIdx>;
        constexpr TIdx xIndex = Dim::value - 1u;

        Vec elementsPerThread(Vec::all(static_cast<TIdx>(1u)));
        Vec threadsPerBlock(Vec::all(static_cast<TIdx>(1u)));
        Vec blocksPerGrid(Vec::all(static_cast<TIdx>(1u)));

        Vec const resultBufferExtent(Vec::all(static_cast<TIdx>(1u)));
        auto resultBuffer(alpaka::mem::buf::alloc<TRed, TIdx>(devAcc, resultBufferExtent));

        // in case n < blockSize, the block reductions only work
        // if the MemAccessPolicy maps the correct values.
        if(n < blockSize) {
            auto resultBuffer(alpaka::mem::buf::alloc<TRed, TIdx>(devAcc, resultBufferExtent));
            WorkDiv dummyWorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
            detail::SmallProblemReduceKernel kernel;
            alpaka::kernel::exec<TAcc>(queue, dummyWorkDiv, kernel, buffer, alpaka::mem::view::getPtrNative(resultBuffer), n, transformFunc, func);
            auto resultView(alpaka::mem::buf::alloc<TRed, TIdx >(devHost, resultBufferExtent));
           // TRed result;
           // alpaka::mem::view::ViewPlainPtr<TDevHost, TRed, Dim, TIdx> resultView{&result, devHost, static_cast<TIdx>(1u)};
            alpaka::mem::view::copy(queue, resultView, resultBuffer, resultBufferExtent);
            alpaka::wait::wait(queue);
            auto result = alpaka::mem::view::getPtrNative(resultView);
            return result[0];
        }


        TIdx gridSize = WorkDivPolicy::template getGridSize<TAcc>(devAcc);

        TIdx maxGridSize = static_cast<TIdx>(
                (((n + 1) / 2) - 1) / static_cast<TIdx>(blockSize) + 1);
        if(gridSize > maxGridSize) {
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

        WorkDiv multiBlockWorkDiv{ blocksPerGrid, threadsPerBlock, elementsPerThread };
        WorkDiv singleBlockWorkDiv{ singleBlocksPerGrid, singleThreadsPerBlock, singleElementsPerThread};

        auto secondPhaseBuffer(alpaka::mem::buf::alloc<TRed, TIdx >(devAcc, sharedMemExtent));

        detail::BlockThreadReduceKernel<blockSize, MemAccessPolicy, TRed> multiBlockKernel, singleBlockKernel;
        // execute kernels
        alpaka::kernel::exec<TAcc>(queue, multiBlockWorkDiv, multiBlockKernel, buffer, alpaka::mem::view::getPtrNative(secondPhaseBuffer), n, transformFunc, func);
        alpaka::kernel::exec<TAcc>(queue, singleBlockWorkDiv, singleBlockKernel, alpaka::mem::view::getPtrNative(secondPhaseBuffer), alpaka::mem::view::getPtrNative(secondPhaseBuffer), gridSize, detail::Identity<TRed>(), func);

        auto resultView(alpaka::mem::buf::alloc<TRed, TIdx >(devHost, resultBufferExtent));
        alpaka::mem::view::copy(queue, resultView, secondPhaseBuffer, resultBufferExtent);

        // wait for result, otherwise the async CPU queue causes a segfault
        alpaka::wait::wait(queue);

        auto result = alpaka::mem::view::getPtrNative(resultView);
        return result[0];
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
     * @param devAcc
     * @param devHost
     * @param queue
     * @param n
     * @param buffer
     * @param func
     * @return
     */
    template<typename TAcc, typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>, typename MemAccessPolicy = vikunja::mem::iterator::MemAccessPolicy<TAcc>, typename TFunc, typename TInputIterator, typename TDevAcc, typename TDevHost, typename TQueue, typename TIdx >
    auto deviceReduce(TDevAcc &devAcc, TDevHost &devHost, TQueue &queue,  TIdx const &n, TInputIterator const &buffer,  TFunc const &func) -> decltype(func(*buffer, *buffer)) {

        using TRed = decltype(func(*buffer, *buffer));
        return deviceTransformReduce<TAcc, WorkDivPolicy, MemAccessPolicy, detail::Identity<TRed>, TFunc, TInputIterator, TDevAcc, TDevHost, TQueue, TIdx>(devAcc, devHost, queue, n, buffer, detail::Identity<TRed>(), func);
    }
}
}
