/* Copyright 2022 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "BaseStrategy.hpp"

#include <alpaka/alpaka.hpp>

namespace vikunja::MemAccess
{
    /**
     * A policy based memory access strategy that splits the data access into chunks. Depending on the memory access
     * policy, these chunks can access the data sequential or in a striding pattern.
     * @tparam MemAccessPolicy The memory access policy to use.
     * @tparam TAcc The alpaka accelerator type.
     * @tparam TIdx The index type
     *
     * The memory access policy should provide three values:
     * - The startIndex of the iterator, which is the first index to use.
     * - The endIndex of the iterator, which is the last index to use.
     * - The stepSize of the iterator, which tells how far the iterator should move.
     */
    template<typename MemAccessPolicy, typename TAcc, typename TIdx>
    class PolicyBasedBlockStrategy : public BaseStrategy<TIdx>
    {
    private:
        TIdx m_step; /**< The step size of this iterator. */
    public:
        /**
         * Create a policy based block iterator
         * @param acc The accelerator type to use.
         * @param problemSize The size of the original iterator.
         * @param blockSize The size of the blocks.
         */
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PolicyBasedBlockStrategy(TAcc const& acc, TIdx problemSize, TIdx blockSize)
            : BaseStrategy<TIdx>(
                MemAccessPolicy::getStartIndex(acc, problemSize, blockSize),
                MemAccessPolicy::getEndIndex(acc, problemSize, blockSize))
            , m_step(MemAccessPolicy::getStepSize(acc, problemSize, blockSize))
        {
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PolicyBasedBlockStrategy(const PolicyBasedBlockStrategy& other) = default;

        //-----------------------------------------------------------------------------
        //! Returns a memory access object with the index set to the last item.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto end() const -> PolicyBasedBlockStrategy
        {
            PolicyBasedBlockStrategy ret(*this);
            ret.m_index = this->m_maximum;
            return ret;
        }

        //-----------------------------------------------------------------------------
        //! Increments the internal index to the next one.
        //!
        //! Returns a reference to the next index.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++() -> PolicyBasedBlockStrategy&
        {
            this->m_index += this->m_step;
            return *this;
        }

        //-----------------------------------------------------------------------------
        //! Returns the current index and increments the internal index to the
        //! next one.
        //!
        //! Returns a reference to the current index.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++(int) -> PolicyBasedBlockStrategy
        {
            auto ret(*this);
            this->m_index += this->m_step;
            return ret;
        }

        //-----------------------------------------------------------------------------
        //! Decrements the internal index to the previous one and returns the this
        //! element.
        //!
        //! Returns a reference to the previous index.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--() -> PolicyBasedBlockStrategy&
        {
            this->m_index -= this->m_step;
            return *this;
        }

        //-----------------------------------------------------------------------------
        //! Returns the current index and decrements the internal pointer to the
        //! previous one.
        //!
        //! Returns a reference to the current index.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--(int) -> PolicyBasedBlockStrategy
        {
            auto ret(*this);
            this->m_index -= this->m_step;
            return ret;
        }

        //-----------------------------------------------------------------------------
        //! Returns the index + a supplied offset.
        //!
        //! \param n The offset.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+(uint64_t n) const -> PolicyBasedBlockStrategy
        {
            auto ret(*this);
            ret.m_index += n * m_step;
            return ret;
        }

        //-----------------------------------------------------------------------------
        //! Returns the index - a supplied offset.
        //!
        //! \param n The offset.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-(uint64_t n) const -> PolicyBasedBlockStrategy
        {
            auto ret(*this);
            ret.m_index -= n * m_step;
            return ret;
        }

        //-----------------------------------------------------------------------------
        //! Addition assignment.
        //!
        //! \param offset The offset.
        //!
        //! Returns the current object offset by the offset.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+=(uint64_t offset) -> PolicyBasedBlockStrategy&
        {
            this->m_index += offset * this->m_step;
            return *this;
        }

        //-----------------------------------------------------------------------------
        //! Substraction assignment.
        //!
        //! \param offset The offset.
        //!
        //! Returns the current object offset by the offset.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-=(uint64_t offset) -> PolicyBasedBlockStrategy&
        {
            this->m_index -= offset * this->m_step;
            return *this;
        }
    };

    namespace policies
    {
        /**
         * A memory policy for the PolicyBlockBasedIterator that provides grid striding memory access.
         */
        struct GridStridingMemAccessPolicy
        {
            template<typename TAcc, typename TIdx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getStartIndex(
                TAcc const& acc,
                TIdx const& problemSize __attribute__((unused)),
                TIdx const& blockSize __attribute__((unused))) -> TIdx const
            {
                constexpr TIdx xIndex = alpaka::Dim<TAcc>::value - 1u;
                return alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[xIndex];
            }

            template<typename TAcc, typename TIdx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getEndIndex(
                TAcc const& acc,
                TIdx const& problemSize,
                TIdx const& blockSize __attribute__((unused))) -> TIdx const
            {
                return problemSize;
            }

            template<typename TAcc, typename TIdx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getStepSize(
                TAcc const& acc,
                TIdx const& problemSize __attribute__((unused)),
                TIdx const& blockSize) -> TIdx const
            {
                constexpr TIdx xIndex = alpaka::Dim<TAcc>::value - 1u;
                auto gridDimension = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[xIndex];
                return gridDimension * blockSize;
            }

            template<typename TAcc, typename TIdx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto isValidThreadResult(
                TAcc const& acc,
                TIdx const& problemSize,
                TIdx const& blockSize __attribute__((unused))) -> bool const
            {
                constexpr TIdx xIndex = alpaka::Dim<TAcc>::value - 1u;
                auto threadIndex = (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[xIndex]);
                return threadIndex < problemSize;
            }

            static constexpr bool isThreadOrderCompliant = true;

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto getName() -> char*
            {
                return const_cast<char*>("GridStridingMemAccessPolicy");
            }
        };

        /**
         * A memory access policy for the PolicyBlockBasedIterator that provides linear memory access.
         */
        struct LinearMemAccessPolicy
        {
            template<typename TAcc, typename TIdx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getStartIndex(
                TAcc const& acc,
                TIdx const& problemSize,
                TIdx const& blockSize) -> TIdx const
            {
                constexpr TIdx xIndex = alpaka::Dim<TAcc>::value - 1u;
                auto gridDimension = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[xIndex];
                auto indexInBlock = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[xIndex];
                auto gridSize = gridDimension * blockSize;
                // TODO: catch overflow
                return (problemSize * indexInBlock) / gridSize;
            }

            template<typename TAcc, typename TIdx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getEndIndex(
                TAcc const& acc,
                TIdx const& problemSize,
                TIdx const& blockSize) -> TIdx const
            {
                constexpr TIdx xIndex = alpaka::Dim<TAcc>::value - 1u;
                auto gridDimension = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[xIndex];
                auto indexInBlock = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[xIndex];
                auto gridSize = gridDimension * blockSize;
                // TODO: catch overflow
                return (problemSize * indexInBlock + problemSize) / gridSize;
            }

            template<typename TAcc, typename TIdx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto getStepSize(
                TAcc const& acc __attribute__((unused)),
                TIdx const& problemSize __attribute__((unused)),
                TIdx const& blockSize __attribute__((unused))) -> TIdx const
            {
                return 1;
            }

            template<typename TAcc, typename TIdx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto isValidThreadResult(
                TAcc const& acc __attribute__((unused)),
                TIdx const& problemSize __attribute__((unused)),
                TIdx const& blockSize __attribute__((unused))) -> bool const
            {
                return true;
            }

            static constexpr bool isThreadOrderCompliant = false;

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto getName() -> char*
            {
                return const_cast<char*>("LinearMemAccessPolicy");
            }
        };
    } // namespace policies

    namespace traits
    {
        /**
         * The memory access policy getter trait by platform.
         * @tparam TAcc The platform type.
         * @tparam TSfinae
         */
        template<typename TPtlf, typename TSfinae = void>
        struct GetMemAccessPolicyByPltf
        {
        };

        /**
         * On cpu, default memory access is linear.
         */
        template<>
        struct GetMemAccessPolicyByPltf<alpaka::PltfCpu>
        {
            using type = policies::LinearMemAccessPolicy;
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        /**
         * On cuda, default memory access is grid striding.
         */
        template<>
        struct GetMemAccessPolicyByPltf<alpaka::PltfUniformCudaHipRt>
        {
            using type = policies::GridStridingMemAccessPolicy;
        };
#endif // (ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

    } // namespace traits

    /**
     * Shortcut to derive memory access policy from accelerator.
     */
    template<typename TAcc>
    using MemAccessPolicy = typename traits::GetMemAccessPolicyByPltf<alpaka::Pltf<alpaka::Dev<TAcc>>>::type;

} // namespace vikunja::MemAccess
