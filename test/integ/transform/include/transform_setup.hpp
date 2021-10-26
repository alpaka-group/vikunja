/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory>

#include <alpaka/alpaka.hpp>
#include <vikunja/test/AlpakaSetup.hpp>

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
                typename TIdx = std::uint64_t>
            class TestSetupBase
                : public vikunja::test::TestAlpakaSetup<TDim, TIdx, alpaka::AccCpuSerial, TAcc, alpaka::Blocking>
            {
            protected:
                using P = vikunja::test::TestAlpakaSetup<TDim, TIdx, alpaka::AccCpuSerial, TAcc, alpaka::Blocking>;
                using Host = typename P::Host;
                using Acc = typename P::Acc;

                using Vec = alpaka::Vec<TDim, TIdx>;
                using BufHost = alpaka::Buf<Host, TData, TDim, TIdx>;
                using BufDev = alpaka::Buf<Acc, TData, TDim, TIdx>;

                std::uint64_t const m_size;
                Vec m_extent;

                BufHost m_host_input1_mem;
                BufDev m_device_input1_mem;

                BufHost m_host_output_mem;
                BufDev m_device_output_mem;

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
                    , m_host_input1_mem(alpaka::allocBuf<TData, TIdx>(P::devHost, m_extent))
                    , m_device_input1_mem(alpaka::allocBuf<TData, TIdx>(P::devAcc, m_extent))
                    , m_host_output_mem(alpaka::allocBuf<TData, TIdx>(P::devHost, m_extent))
                    , m_device_output_mem(alpaka::allocBuf<TData, TIdx>(P::devAcc, m_extent))
                {
                }

                TData* get_host_input1_mem_ptr()
                {
                    return alpaka::getPtrNative(m_host_input1_mem);
                }

                TData* get_host_output_mem_ptr()
                {
                    return alpaka::getPtrNative(m_host_output_mem);
                }
            };


        } // namespace transform
    } // namespace test
} // namespace vikunja
