/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <vikunja/test/AlpakaSetup.hpp>

#include <alpaka/alpaka.hpp>

#include <memory>

namespace vikunja
{
    namespace test
    {
        namespace transform
        {
            template<
                typename TDim,
                template<typename, typename>
                class TAcc,
                typename TData,
                typename TDataResult,
                typename TIdx = std::uint64_t>
            class TestSetupBase
                : public vikunja::test::TestAlpakaSetup<TDim, TIdx, alpaka::AccCpuSerial, TAcc, alpaka::Blocking>
            {
            protected:
                using Base = vikunja::test::TestAlpakaSetup<TDim, TIdx, alpaka::AccCpuSerial, TAcc, alpaka::Blocking>;
                using Host = typename Base::Host;
                using Acc = typename Base::Acc;

                using Vec = alpaka::Vec<TDim, TIdx>;
                using BufHost = alpaka::Buf<Host, TData, TDim, TIdx>;
                using BufDev = alpaka::Buf<Acc, TData, TDim, TIdx>;
                using BufHostResult = alpaka::Buf<Host, TDataResult, TDim, TIdx>;
                using BufDevResult = alpaka::Buf<Acc, TDataResult, TDim, TIdx>;

                std::uint64_t const m_size;
                Vec m_extent;

                BufHost m_host_input1_mem;
                BufDev m_device_input1_mem;

                BufHostResult m_host_output_mem;
                BufDevResult m_device_output_mem;

            private:
                Vec calculate_extends(std::uint64_t const size)
                {
                    Vec extent = Vec::all(static_cast<TIdx>(1));
                    extent[TDim::value - 1u] = size;
                    return extent;
                }

            public:
                TestSetupBase(uint64_t const memSize)
                    : m_size(memSize)
                    , m_extent(calculate_extends(memSize))
                    , m_host_input1_mem(alpaka::allocBuf<TData, TIdx>(Base::devHost, m_extent))
                    , m_device_input1_mem(alpaka::allocBuf<TData, TIdx>(Base::devAcc, m_extent))
                    , m_host_output_mem(alpaka::allocBuf<TDataResult, TIdx>(Base::devHost, m_extent))
                    , m_device_output_mem(alpaka::allocBuf<TDataResult, TIdx>(Base::devAcc, m_extent))
                {
                }

                TData* get_host_input1_mem_ptr()
                {
                    return alpaka::getPtrNative(m_host_input1_mem);
                }

                TDataResult* get_host_output_mem_ptr()
                {
                    return alpaka::getPtrNative(m_host_output_mem);
                }
            };

        } // namespace transform
    } // namespace test
} // namespace vikunja
