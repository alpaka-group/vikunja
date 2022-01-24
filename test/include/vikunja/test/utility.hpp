/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <sstream>

#define REQUIRE_MESSAGE(cond, msg)                                                                                    \
    do                                                                                                                \
    {                                                                                                                 \
        INFO(msg);                                                                                                    \
        REQUIRE(cond);                                                                                                \
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

            using MemAccess = vikunja::mem::iterator::MemAccessPolicy<Acc>;
            strs << "MemAccessPolicy: " << MemAccess::getName() << "\n";

            return strs.str();
        }

    } // namespace test
} // namespace vikunja
