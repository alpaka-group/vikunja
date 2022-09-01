#pragma once

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace vikunja::concept
{
    struct AccNotEnabledTag
    {
        static std::string get_name()
        {
            return "AccNotEnabledTag";
        }
    };
    struct AccGpuCudaRtTag
    {
        static std::string get_name()
        {
            return "AccGpuCudaRtTag";
        }
    };
    struct AccGpuHipRtTag
    {
        static std::string get_name()
        {
            return "AccGpuHipRtTag";
        }
    };
    struct AccCpuThreadsTag
    {
        static std::string get_name()
        {
            return "AccCpuThreadsTag";
        }
    };
    struct AccCpuFibersTag
    {
        static std::string get_name()
        {
            return "AccCpuFibersTag";
        }
    };
    struct AccCpuOmp2ThreadsTag
    {
        static std::string get_name()
        {
            return "AccCpuOmp2ThreadsTag";
        }
    };
    struct AccCpuOmp2BlocksTag
    {
        static std::string get_name()
        {
            return "AccCpuOmp2BlocksTag";
        }
    };
    struct AccOmp5Tag
    {
        static std::string get_name()
        {
            return "AccOmp5Tag";
        }
    };
    struct AccCpuTbbBlocksTag
    {
        static std::string get_name()
        {
            return "AccCpuTbbBlocksTag";
        }
    };
    struct AccCpuSerialTag
    {
        static std::string get_name()
        {
            return "AccCpuSerialTag";
        }
    };

    template<typename TAcc>
    struct AccToTag
    {
        using Tag = vikunja::concept::AccNotEnabledTag;
    };

    template<typename TTag, typename TDim, typename TIdx>
    struct TagToAcc;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccGpuCudaRt<TDim, TIdx>>
    {
        using Tag = vikunja::concept::AccGpuCudaRtTag;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<vikunja::concept::AccGpuCudaRtTag, TDim, TIdx>
    {
        using Acc = alpaka::AccGpuCudaRt<TDim, TIdx>;
    };
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccCpuSerial<TDim, TIdx>>
    {
        using Tag = vikunja::concept::AccCpuSerialTag;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<vikunja::concept::AccCpuSerialTag, TDim, TIdx>
    {
        using Acc = alpaka::AccCpuSerial<TDim, TIdx>;
    };
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccCpuOmp2Threads<TDim, TIdx>>
    {
        using Tag = vikunja::concept::AccCpuOmp2ThreadsTag;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<vikunja::concept::AccCpuOmp2ThreadsTag, TDim, TIdx>
    {
        using Acc = alpaka::AccCpuOmp2Threads<TDim, TIdx>;
    };
#endif

#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccCpuOmp2Blocks<TDim, TIdx>>
    {
        using Tag = vikunja::concept::AccCpuOmp2BlocksTag;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<vikunja::concept::AccCpuOmp2BlocksTag, TDim, TIdx>
    {
        using Acc = alpaka::AccCpuOmp2Blocks<TDim, TIdx>;
    };
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccGpuHipRt<TDim, TIdx>>
    {
        using Tag = vikunja::concept::AccGpuHipRtTag;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<vikunja::concept::AccGpuHipRtTag, TDim, TIdx>
    {
        using Acc = alpaka::AccGpuHipRt<TDim, TIdx>;
    };
#endif

    template<typename TAcc, typename TTag>
    constexpr bool acc_tag_match()
    {
        return std::is_same_v<typename vikunja::concept::AccToTag<TAcc>::Tag, TTag>;
    }

} // namespace vikunja::concept
