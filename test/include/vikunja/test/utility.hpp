/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <sstream>
#include <alpaka/alpaka.hpp>
#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>

namespace vikunja
{
    namespace test
    {
        template<typename TDim, typename TData>
        inline std::string print_acc_info(TData const size)
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
