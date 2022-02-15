/* Copyright 2022 Hauke Mewes, Jonas Schenke, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

namespace vikunja::MemAccess
{
    //! Base class to implement memory access strategy.
    //!
    //! \tparam TIdx Index type
    template<typename TIdx>
    class BaseStrategy
    {
    protected:
        TIdx m_index;
        TIdx const m_maximum;

    public:
        //-----------------------------------------------------------------------------
        //! Constructor.
        //!
        //! \param index The index.
        //! \param maximum value of the index (not inclusive).
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE BaseStrategy(TIdx const index, TIdx const maximum)
            : m_index(index)
            , m_maximum(maximum)
        {
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE BaseStrategy(const BaseStrategy& other) = default;

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr friend auto operator==(
            BaseStrategy const& st,
            const BaseStrategy& other) -> bool
        {
            return (st.m_index == other.m_index) && (st.m_maximum == other.m_maximum);
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr friend auto operator!=(
            BaseStrategy const& st,
            const BaseStrategy& other) -> bool
        {
            return !operator==(st, other);
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr friend auto operator<(
            BaseStrategy const& st,
            const BaseStrategy& other) -> bool
        {
            return st.m_index < other.m_index;
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr friend auto operator>(
            BaseStrategy const& st,
            const BaseStrategy& other) -> bool
        {
            return st.m_index > other.m_index;
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr friend auto operator<=(
            BaseStrategy const& st,
            const BaseStrategy& other) -> bool
        {
            return st.m_index <= other.m_index;
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr friend auto operator>=(
            BaseStrategy const& st,
            const BaseStrategy& other) -> bool
        {
            return st.m_index >= other.m_index;
        }

        //-----------------------------------------------------------------------------
        //! Get the current index.
        //!
        //! Returns a const reference to the current index.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator*() const -> TIdx const&
        {
            return m_index;
        }

        //-----------------------------------------------------------------------------
        //! Set the current index
        //!
        //! Returns a reference to the current index.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator*() -> TIdx&
        {
            return m_index;
        }
    };

} // namespace vikunja::MemAccess
