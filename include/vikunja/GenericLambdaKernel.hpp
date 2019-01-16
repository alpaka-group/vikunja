//
// Created by mewes30 on 17.10.18.
//

#pragma once

#include "alpaka/alpaka.hpp"

namespace vikunja {

    /**
     * Class that provides an alpaka kernel which can use an arbitrary functor to avoid the same dimension
     * calculations in each of those.
     * @tparam TOp Type of the functor.
     * @tparam provideAcc Whether the accelerator itself should be provided as an argument.
     */
    template<typename TOp, bool provideAcc = false>
    class GenericLambdaKernel {

    };

    /**
     * Specialization that does not provide the accelerator as an argument.
     * @tparam TOp Type of the functor.
     */
    template<typename TOp>
    class GenericLambdaKernel<TOp, false> {
    private:
        /**
         * Functor to call.
         */
        const TOp operation;
    public:

        /**
         * Construct a kernel from the provided functor.
         * @param op Functor to call.
         */
        explicit GenericLambdaKernel(const TOp &op) : operation(op) {}

        /**
         * Overload of the application operator that implements the functor kernel.
         * @tparam TAcc Accelerator type.
         * @tparam TIdx Index type.
         * @tparam TArgs Argument types.
         * @param acc Accelerator.
         * @param numElements Total number of processed elements.
         * @param args Arguments.
         */
        template<typename TAcc, typename TIdx, typename... TArgs>
        ALPAKA_FN_ACC auto operator()(TAcc const &acc, TIdx const &numElements, TArgs & ...args) const -> void {
            TIdx const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
            TIdx const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
            TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

            if(threadFirstElemIdx < numElements)
            {

                TIdx const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
                TIdx const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

                for(TIdx i(threadFirstElemIdx); i <  threadLastElemIdxClipped; ++i)
                {
                    operation(i, args...);
                }
            }
        }
    };

    /**
     * Specialization that does provide the accelerator as an argument.
     * @tparam TOp Type of the functor.
     */
    template<typename TOp>
    class GenericLambdaKernel<TOp, true> {
    private:
        /**
         * Functor to call.
         */
        const TOp operation;
    public:

        /**
         * Construct a kernel from the provided functor.
         * @param op Functor to call.
         */
        explicit GenericLambdaKernel(const TOp &op) : operation(op) {}

        /**
         * Overload of the application operator that implements the functor kernel.
         * @tparam TAcc Accelerator type.
         * @tparam TIdx Index type.
         * @tparam TArgs Argument types.
         * @param acc Accelerator.
         * @param numElements Total number of processed elements.
         * @param args Arguments.
         */
        template<typename TAcc, typename TIdx, typename... TArgs>
        ALPAKA_FN_ACC auto operator()(TAcc const &acc, TIdx const &numElements, TArgs & ...args) const -> void {
            TIdx const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
            TIdx const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
            TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

            if(threadFirstElemIdx < numElements)
            {

                TIdx const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
                TIdx const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

                for(TIdx i(threadFirstElemIdx); i <  threadLastElemIdxClipped; ++i)
                {
                    operation(acc, i, args...);
                }
            }
        }
    };
}
