/* Copyright 2022 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/concept/memVisibility.hpp>
#include <vikunja/test/utility.hpp>

#include <alpaka/alpaka.hpp>

#include <type_traits>

#include <catch2/catch.hpp>


TEST_CASE("GetVisibilityTypeFromAcc", "[buffer]")
{
    using Dim = alpaka::DimInt<1>;
    using Idx = int;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using CMP = vikunja::concept::CUDAMemVisible;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using CMP = vikunja::concept::HIPMemVisible;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)                     \
    || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)                    \
    || defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
    using CMP = vikunja::concept::CPUMemVisible;
#endif

    INFO(("Testing accelerator: " + alpaka::getAccName<Acc>() + "\n"));
    INFO((std::string("expected buffer type: ") + vikunja::test::print_buffer_type<CMP>()));

    REQUIRE_MESSAGE(
        (std::is_same_v<CMP, vikunja::concept::get_mem_visibility_type<Acc>>),
        std::string("got buffer type: ")
            + vikunja::test::print_buffer_type<vikunja::concept::get_mem_visibility_type<Acc>>());
}

TEST_CASE("IsVisibilityTypeSupportAcc", "[buffer]")
{
    using Dim = alpaka::DimInt<1>;
    using Idx = int;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using CMP = vikunja::concept::CUDAMemVisible;
    using WrongCMP1 = vikunja::concept::CPUMemVisible;
    using WrongCMP2 = vikunja::concept::HIPMemVisible;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using CMP = vikunja::concept::HIPMemVisible;
    using WrongCMP1 = vikunja::concept::CPUMemVisible;
    using WrongCMP2 = vikunja::concept::CUDAMemVisible;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)                     \
    || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)                    \
    || defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
    using CMP = vikunja::concept::CPUMemVisible;
    using WrongCMP1 = vikunja::concept::HIPMemVisible;
    using WrongCMP2 = vikunja::concept::CUDAMemVisible;
#endif

    REQUIRE((vikunja::concept::is_mem_visibility_type_support_acc<Acc, CMP>()));
    REQUIRE_FALSE(vikunja::concept::is_mem_visibility_type_support_acc<Acc, WrongCMP1>());
    REQUIRE_FALSE(vikunja::concept::is_mem_visibility_type_support_acc<Acc, WrongCMP2>());
}
