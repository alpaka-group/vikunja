/* Copyright 2022 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <vikunja/access/BlockStrategy.hpp>
#include <vikunja/concept/memVisibility.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <sstream>

#define REQUIRE_MESSAGE(cond, msg)                                                                                    \
    do                                                                                                                \
    {                                                                                                                 \
        INFO(msg);                                                                                                    \
        REQUIRE(cond);                                                                                                \
    } while((void) 0, 0)

#define REQUIRE_FALSE_MESSAGE(cond, msg)                                                                              \
    do                                                                                                                \
    {                                                                                                                 \
        INFO(msg);                                                                                                    \
        REQUIRE_FALSE(cond);                                                                                          \
    } while((void) 0, 0)

namespace vikunja
{
    namespace test
    {
        template<typename TDim>
        inline std::string print_acc_info(std::size_t const size)
        {
            std::stringstream strs;

            using Acc = alpaka::ExampleDefaultAcc<TDim, std::uint64_t>;
            strs << "Testing accelerator: " << alpaka::getAccName<Acc>() << " with size: " << size << "\n";

            using MemAccess = vikunja::MemAccess::MemAccessPolicy<Acc>;
            strs << "MemAccessPolicy: " << MemAccess::getName() << "\n";

            return strs.str();
        }

        template<typename TBufferType>
        constexpr inline char const* print_buffer_type()
        {
            if(std::is_same_v<TBufferType, vikunja::concept::CUDAMemVisible>)
            {
                return "CUDAMemVisible";
            }
            else if(std::is_same_v<TBufferType, vikunja::concept::HIPMemVisible>)
            {
                return "HIPMemVisible";
            }
            else if(std::is_same_v<TBufferType, vikunja::concept::CPUMemVisible>)
            {
                return "CPUMemVisible";
            }
            else
            {
                return "unknown buffer";
            }
        }

    } // namespace test
} // namespace vikunja
