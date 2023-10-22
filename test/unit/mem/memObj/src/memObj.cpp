/* Copyright 2022 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include <vikunja/mem/memObj.hpp>
#include <vikunja/test/utility.hpp>

#include <type_traits>

#include <catch2/catch.hpp>


TEST_CASE("memObj visibility", "[memObj]")
{
    using Dim = alpaka::DimInt<1>;
    using Idx = int;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using ExpectedVisibilityType = vikunja::concept::CUDAMemVisible;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using ExpectedVisibilityType = vikunja::concept::HIPBuffer;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)                     \
    || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)                    \
    || defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
    using ExpectedVisibilityType = vikunja::concept::CPUMemVisible;
#endif

    vikunja::MemObj<Acc> memObj;

    REQUIRE((std::is_same_v<decltype(memObj)::Acc, Acc>) );
    REQUIRE((std::is_same_v<decltype(memObj)::MemVisibility, ExpectedVisibilityType>) );
}
