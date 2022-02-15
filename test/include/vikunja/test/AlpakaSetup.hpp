/* Copyright 2022 Hauke Mewes, Simeon Ehrig
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
    namespace test
    {
        template<
            typename TDim,
            typename TIdx,
            template<class, class>
            class THost,
            template<class, class>
            class TAcc,
            typename TQueue>
        struct TestAlpakaSetup
        {
        public:
            using Dim = TDim;
            using Idx = TIdx;
            using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

            using Host = THost<Dim, Idx>;
            using PltfHost = alpaka::Pltf<Host>;
            using DevHost = alpaka::Dev<PltfHost>;

            using Acc = TAcc<Dim, Idx>;
            using PltfAcc = alpaka::Pltf<Acc>;
            using DevAcc = alpaka::Dev<PltfAcc>;

            using QueueAcc = alpaka::Queue<DevAcc, TQueue>;

            DevAcc devAcc;
            DevHost devHost;
            QueueAcc queueAcc;

            TestAlpakaSetup()
                : devAcc{alpaka::getDevByIdx<PltfAcc>(0u)}
                , devHost{alpaka::getDevByIdx<PltfHost>(0u)}
                , queueAcc{devAcc}
            {
            }

            /**
             * @brief Allocate 1D memory on the host.
             *
             * @tparam TData Type of the memory.
             * @param size Size of the memory.
             * @return auto Alpaka memory buffer on the host.
             */
            template<typename TData>
            auto allocHost(Idx size)
            {
                using Vec = alpaka::Vec<alpaka::DimInt<1>, Idx>;
                return alpaka::allocBuf<TData, Idx>(devHost, Vec::all(size));
            }

            /**
             * @brief Allocate ND memory on the host.
             *
             * @tparam TData Type of the memory.
             * @tparam TExtentDim alpaka::Vec<N, Idx>
             * @param size Vector with the sizes for each memory dimension.
             * @return auto Alpaka memory buffer on the host.
             */
            template<typename TData, typename TExtentDim>
            auto allocHost(alpaka::Vec<TExtentDim, Idx> size)
            {
                return alpaka::allocBuf<TData, Idx>(devHost, size);
            }

            /**
             * @brief Allocate 1D memory on the device.
             *
             * @tparam TData Type of the memory.
             * @param size Size of the memory.
             * @return auto Alpaka memory buffer on the device.
             */
            template<typename TData>
            auto allocDev(Idx size)
            {
                using Vec = alpaka::Vec<alpaka::DimInt<1>, Idx>;
                return alpaka::allocBuf<TData, Idx>(devAcc, Vec::all(size));
            }

            /**
             * @brief Allocate ND memory on the device.
             *
             * @tparam TData Type of the memory.
             * @tparam TExtentDim alpaka::Vec<N, Idx>
             * @param size Vector with the sizes for each memory dimension.
             * @return auto Alpaka memory buffer on the device.
             */
            template<typename TData, typename TExtentDim>
            auto allocDev(alpaka::Vec<TExtentDim, Idx> size)
            {
                return alpaka::allocBuf<TData, Idx>(devAcc, size);
            }
        };
    } // namespace test
} // namespace vikunja
