/* Copyright 2022 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/concept/acc.hpp>
#include <vikunja/test/utility.hpp>

#include <alpaka/alpaka.hpp>

#include <type_traits>

#include <catch2/catch.hpp>

TEST_CASE("AccTags", "[acc]")
{
    using Dim = alpaka::DimInt<1>;
    using Idx = int;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    if constexpr(std::is_same_v<alpaka::AccCpuSerial<Dim, Idx>, Acc>)
    {
        REQUIRE_MESSAGE(
            (std::is_same_v<typename vikunja::concept::AccToTag<Acc>::Tag, vikunja::concept::AccCpuSerialTag>),
            std::string("acc_tag_match() returns tag: ") + vikunja::concept::AccToTag<Acc>::Tag::get_name());
        // STATIC_REQUIRE does not work, because nvcc resolves both branches independent of the result of the if
        // condition. This causes a compile time error.
        REQUIRE((std::is_same_v<
                 typename vikunja::concept::TagToAcc<vikunja::concept::AccCpuSerialTag, Dim, Idx>::Acc,
                 Acc>) );
        REQUIRE((vikunja::concept::acc_tag_match<Acc, vikunja::concept::AccCpuSerialTag>()));
    }
    else
    {
        REQUIRE_FALSE((vikunja::concept::acc_tag_match<Acc, vikunja::concept::AccCpuSerialTag>()));
    }
#else
    REQUIRE_FALSE((vikunja::concept::acc_tag_match<Acc, vikunja::concept::AccCpuSerialTag>()));
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    if constexpr(std::is_same_v<alpaka::AccGpuCudaRt<Dim, Idx>, Acc>)
    {
        REQUIRE_MESSAGE(
            (std::is_same_v<typename vikunja::concept::AccToTag<Acc>::Tag, vikunja::concept::AccGpuCudaRtTag>),
            std::string("acc_tag_match() returns tag: ") + vikunja::concept::AccToTag<Acc>::Tag::get_name());
        REQUIRE((std::is_same_v<
                 typename vikunja::concept::TagToAcc<vikunja::concept::AccGpuCudaRtTag, Dim, Idx>::Acc,
                 Acc>) );
        REQUIRE((vikunja::concept::acc_tag_match<Acc, vikunja::concept::AccGpuCudaRtTag>()));
    }
    else
    {
        REQUIRE_FALSE((vikunja::concept::acc_tag_match<Acc, vikunja::concept::AccGpuCudaRtTag>()));
    }
#else
    REQUIRE_FALSE((vikunja::concept::acc_tag_match<Acc, vikunja::concept::AccGpuCudaRtTag>()));
#endif
}

template<typename Tag>
std::string specialized_function()
{
    return "Generic";
}

template<>
std::string specialized_function<vikunja::concept::AccCpuSerialTag>()
{
    return "Serial";
}

template<>
std::string specialized_function<vikunja::concept::AccGpuCudaRtTag>()
{
    return "CUDA";
}

TEST_CASE("is_specialized", "[acc]")
{
    using Dim = alpaka::DimInt<1>;
    using Idx = int;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

    std::string expected_result = "Generic";

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    if constexpr(std::is_same_v<alpaka::AccCpuSerial<Dim, Idx>, Acc>)
    {
        expected_result = "Serial";
    }
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    if constexpr(std::is_same_v<alpaka::AccGpuCudaRt<Dim, Idx>, Acc>)
    {
        expected_result = "CUDA";
    }
#endif

    REQUIRE_MESSAGE(
        (specialized_function<typename vikunja::concept::AccToTag<Acc>::Tag>() == expected_result),
        "is_specialized<Acc>() returns: " + specialized_function<typename vikunja::concept::AccToTag<Acc>::Tag>());
}
