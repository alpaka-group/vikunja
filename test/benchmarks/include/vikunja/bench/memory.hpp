/* Copyright 2022 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <vector>

namespace vikunja::bench
{
    template<typename TData>
    class IotaFunctor
    {
    private:
        TData const m_begin;
        TData const m_increment;

    public:
        //! Iota functor for generic data types.
        //!
        //! \tparam TData Type of each element
        //! \param init Value of the first element.
        //! \param increment Distance between two elements.
        IotaFunctor(TData const init, TData const increment) : m_begin(init), m_increment(increment)
        {
        }

        //! Writes the result of `init + index * increment` to each element of the output vector.
        //!
        //! \tparam TAcc The accelerator environment to be executed on.
        //! \tparam TElem The element type.
        //! \param acc The accelerator to be executed on.
        //! \param output The destination vector.
        //! \param numElements The number of elements.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TIdx>
        ALPAKA_FN_ACC auto operator()(TAcc const& acc, TData* const output, TIdx const& numElements) const -> void
        {
            static_assert(alpaka::Dim<TAcc>::value == 1, "The VectorAddKernel expects 1-dimensional indices!");

            TIdx const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
            TIdx const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
            TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

            if(threadFirstElemIdx < numElements)
            {
                // Calculate the number of elements for this thread.
                // The result is uniform for all but the last thread.
                TIdx const threadLastElemIdx(threadFirstElemIdx + threadElemExtent);
                TIdx const threadLastElemIdxClipped(alpaka::math::min(acc, numElements, threadLastElemIdx));

                for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
                {
                    output[i] = m_begin + static_cast<TData>(i) * m_increment;
                }
            }
        }
    };


    //! Allocates memory and initializes each value with `init + index * increment`,
    //! where index is the position in the output vector. The allocation is done with `setup.devAcc`.
    //!
    //! \tparam TData Data type of the memory buffer.
    //! \tparam TSetup Fully specialized type of `vikunja::test::TestAlpakaSetup`.
    //! \tparam Type of the extent.
    //! \tparam TBuf Type of the alpaka memory buffer.
    //! \param setup Instance of `vikunja::test::TestAlpakaSetup`. `setup.devAcc` and `setup.queueDev` are used
    //! for allocation and initialization of the the memory.
    //! \param extent Size of the memory buffer. Needs to be 1 dimensional.
    //! \param init Value of the first element. Depending on TData, it can be negative.
    //! \param increment Distance between two elements of the vector. If the value is negative, the value of an
    //! element is greater than its previous element.
    template<
        typename TData,
        typename TSetup,
        typename TExtent,
        typename TBuf = alpaka::Buf<typename TSetup::DevAcc, TData, alpaka::DimInt<1u>, typename TSetup::Idx>>
    TBuf allocate_mem_iota(
        TSetup& setup,
        TExtent const& extent,
        TData const init = TData{0},
        TData const increment = TData{1})
    {
        // TODO: test also 2 and 3 dimensional memory
        static_assert(TExtent::Dim::value == 1);

        // TODO: optimize utilization for CPU backends
        typename TSetup::Idx const elementsPerThread = 1;
        typename TSetup::Idx linSize = extent.prod();

        TBuf devMem(alpaka::allocBuf<TData, typename TSetup::Idx>(setup.devAcc, extent));

        alpaka::WorkDivMembers<typename TSetup::Dim, typename TSetup::Idx> const workDiv(
            alpaka::getValidWorkDiv<typename TSetup::Acc>(
                setup.devAcc,
                extent,
                elementsPerThread,
                false,
                alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

        IotaFunctor iotaFunctor(init, increment);

        alpaka::exec<typename TSetup::Acc>(
            setup.queueAcc,
            workDiv,
            iotaFunctor,
            alpaka::getPtrNative(devMem),
            linSize);

        return devMem;
    }

    template<typename TData>
    class ConstantInitFunctor
    {
    private:
        TData const m_constant;

    public:
        //! Functor to write a constant value into each element of a vector.
        //!
        //! \tparam TData Type of each element
        //! \param constant Value to which all elements are set.
        ConstantInitFunctor(TData const constant) : m_constant(constant)
        {
        }

        //! Writes the constant to each element of the output vector.
        //!
        //! \tparam TAcc The accelerator environment to be executed on.
        //! \tparam TElem The element type.
        //! \param acc The accelerator to be executed on.
        //! \param output The destination vector.
        //! \param numElements The number of elements.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TIdx>
        ALPAKA_FN_ACC auto operator()(TAcc const& acc, TData* const output, TIdx const& numElements) const -> void
        {
            static_assert(alpaka::Dim<TAcc>::value == 1, "The VectorAddKernel expects 1-dimensional indices!");

            TIdx const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
            TIdx const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
            TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

            if(threadFirstElemIdx < numElements)
            {
                // Calculate the number of elements for this thread.
                // The result is uniform for all but the last thread.
                TIdx const threadLastElemIdx(threadFirstElemIdx + threadElemExtent);
                TIdx const threadLastElemIdxClipped(alpaka::math::min(acc, numElements, threadLastElemIdx));

                for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
                {
                    output[i] = m_constant;
                }
            }
        }
    };

    //! Allocates memory and initializes each value with a constant value.
    //! The allocation is done with `setup.devAcc`.
    //!
    //! \tparam TData Data type of the memory buffer.
    //! \tparam TSetup Fully specialized type of `vikunja::test::TestAlpakaSetup`.
    //! \tparam Type of the extent.
    //! \tparam TBuf Type of the alpaka memory buffer.
    //! \param setup Instance of `vikunja::test::TestAlpakaSetup`. `setup.devAcc` and `setup.queueDev` are used
    //! for allocation and initialization of the the memory.
    //! \param extent Size of the memory buffer. Needs to be 1 dimensional.
    //! \param constant Value of the constant.
    template<
        typename TData,
        typename TSetup,
        typename TExtent,
        typename TBuf = alpaka::Buf<typename TSetup::DevAcc, TData, alpaka::DimInt<1u>, typename TSetup::Idx>>
    TBuf allocate_mem_constant(TSetup& setup, TExtent const& extent, TData const constant)
    {
        // TODO: test also 2 and 3 dimensional memory
        static_assert(TExtent::Dim::value == 1);

        // TODO: optimize utilization for CPU backends
        typename TSetup::Idx const elementsPerThread = 1;
        typename TSetup::Idx linSize = extent.prod();

        TBuf devMem(alpaka::allocBuf<TData, typename TSetup::Idx>(setup.devAcc, extent));

        alpaka::WorkDivMembers<typename TSetup::Dim, typename TSetup::Idx> const workDiv(
            alpaka::getValidWorkDiv<typename TSetup::Acc>(
                setup.devAcc,
                extent,
                elementsPerThread,
                false,
                alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

        ConstantInitFunctor constantInitFunctor(constant);

        alpaka::exec<typename TSetup::Acc>(
            setup.queueAcc,
            workDiv,
            constantInitFunctor,
            alpaka::getPtrNative(devMem),
            linSize);

        return devMem;
    }
} // namespace vikunja::bench
