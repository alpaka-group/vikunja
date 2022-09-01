#pragma once

//#include "acc.hpp"

#include <vikunja/concept/acc.hpp>

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace vikunja::concept
{
    struct CPUMemVisible
    {
    };
    struct CUDAMemVisible
    {
    };
    struct HIPMemVisible
    {
    };


    template<typename TAcc, typename = void>
    struct GetMemVisibiltyTypeImpl;

    template<typename TAcc>
    struct GetMemVisibiltyTypeImpl<
        TAcc,
        std::enable_if_t<
            vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccCpuSerialTag>()
            || vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccCpuOmp2ThreadsTag>()
            || vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccCpuOmp2BlocksTag>()>>
    {
        using type = vikunja::concept::CPUMemVisible;
    };

    template<typename TAcc>
    struct GetMemVisibiltyTypeImpl<
        TAcc,
        std::enable_if_t<vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccGpuCudaRtTag>()>>
    {
        using type = vikunja::concept::CUDAMemVisible;
    };

    template<typename TAcc>
    struct GetMemVisibiltyTypeImpl<
        TAcc,
        std::enable_if_t<vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccGpuHipRtTag>()>>
    {
        using type = vikunja::concept::HIPMemVisible;
    };


    template<typename TAcc>
    using get_mem_visibility_type = typename GetMemVisibiltyTypeImpl<TAcc>::type;


    template<typename TAcc, typename TBufferType>
    constexpr inline bool is_mem_visibility_type_support_acc()
    {
        if constexpr(
            std::is_same_v<
                TBufferType,
                vikunja::concept::
                    CPUMemVisible> && (vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccCpuSerialTag>() || vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccCpuOmp2BlocksTag>() || vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccCpuOmp2ThreadsTag>()))
        {
            return true;
        }

        if constexpr(
            std::is_same_v<
                TBufferType,
                vikunja::concept::
                    CUDAMemVisible> && vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccGpuCudaRtTag>())
        {
            return true;
        }

        if constexpr(
            std::is_same_v<
                TBufferType,
                vikunja::concept::
                    HIPMemVisible> && vikunja::concept::acc_tag_match<TAcc, vikunja::concept::AccGpuHipRtTag>())
        {
            return true;
        }

        return false;
    }

} // namespace vikunja::concept
