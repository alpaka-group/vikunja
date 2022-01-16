/* Copyright 2021 Victor
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <iterator>

// check for spaceship support
#if defined __has_include
#    if __has_include(<version>)
#        include <version>
#    endif
#    if __has_include(<compare>)
#        include <compare>
#        if defined(__cpp_lib_three_way_comparison) && defined(__cpp_impl_three_way_comparison)
#            define USESPACESHIP
#        endif
#    endif
#endif

#if __has_cpp_attribute(nodiscard)
#    define NODISCARD [[nodiscard]]
#else
#    define NODISCARD
#endif

namespace vikunja
{
    namespace mem
    {
        namespace iterator
        {
            /**
             * @brief A zip iterator that takes multiple input sequences and yields a sequence of tuples
             * @tparam IteratorTuple The type of the data
             * @tparam IdxType The type of the index
             */
            template<typename IteratorTuple, typename IdxType = int64_t>
            class ZipIterator
            {
            public:
                // Need all 5 of these types for iterator_traits
                using reference = IteratorTuple&;
                using value_type = IteratorTuple;
                using pointer = IteratorTuple*;
                using difference_type = IdxType;
                using iterator_category = std::random_access_iterator_tag;

                /**
                 * @brief Constructor for the ZipIterator
                 * @param iteratorTuple The tuple to initialize the iterator with
                 * @param idx The index for the iterator, default 0
                 */
                ZipIterator(IteratorTuple iteratorTuple, const IdxType& idx = static_cast<IdxType>(0))
                    : m_iteratorTuple(iteratorTuple)
                    , m_index(idx)
                {
                    if (idx != 0)
                        for_each(m_iteratorTuple, [idx](auto &x) { x += idx; });
                }

                /**
                 * @brief Dereference operator to receive the stored value
                 */
                NODISCARD ALPAKA_FN_INLINE IteratorTuple& operator*()
                {
                    return m_iteratorTuple;
                }

                /**
                 * @brief Index operator to get stored value at some given offset from this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE IteratorTuple operator[](const IdxType idx)
                {
                    IteratorTuple tmp = m_iteratorTuple;
                    IdxType indexDiff = idx - m_index;
                    for_each(tmp, [indexDiff](auto &x) { x += indexDiff; });
                    return tmp;
                }

#pragma region arithmeticoperators
                /**
                 * @brief Prefix increment operator
                 */
                NODISCARD ALPAKA_FN_INLINE ZipIterator& operator++()
                {
                    ++m_index;
                    for_each(m_iteratorTuple, [](auto &x) { ++x; });
                    return *this;
                }

                /**
                 * @brief Postfix increment operator
                 * @note Use prefix increment operator instead if possible to avoid copies
                 */
                ALPAKA_FN_INLINE ZipIterator operator++(int)
                {
                    ZipIterator tmp = *this;
                    ++m_index;
                    for_each(m_iteratorTuple, [](auto &x) { ++x; });
                    return tmp;
                }

                /**
                 * @brief Prefix decrement operator
                 */
                NODISCARD ALPAKA_FN_INLINE ZipIterator& operator--()
                {
                    --m_index;
                    for_each(m_iteratorTuple, [](auto &x) { --x; });
                    return *this;
                }

                /**
                 * @brief Postfix decrement operator
                 * @note Use prefix decrement operator instead if possible to avoid copies
                 */
                ALPAKA_FN_INLINE ZipIterator operator--(int)
                {
                    ZipIterator tmp = *this;
                    --m_index;
                    for_each(m_iteratorTuple, [](auto &x) { --x; });
                    return tmp;
                }

                /**
                 * @brief Add an index to this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE ZipIterator operator+(const IdxType idx)
                {
                    IteratorTuple tmp = m_iteratorTuple;
                    IdxType indexDiff = m_index;
                    for_each(tmp, [indexDiff](auto &x) { x -= indexDiff; });
                    return ZipIterator(tmp, m_index + idx);
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE ZipIterator operator-(const IdxType idx)
                {
                    IteratorTuple tmp = m_iteratorTuple;
                    IdxType indexDiff = m_index;
                    for_each(tmp, [indexDiff](auto &x) { x -= indexDiff; });
                    return ZipIterator(tmp, m_index - idx);
                }

                /**
                 * @brief Add an index to this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE ZipIterator& operator+=(const IdxType idx)
                {
                    m_index += idx;
                    for_each(m_iteratorTuple, [idx](auto &x) { x += idx; });
                    return *this;
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE ZipIterator& operator-=(const IdxType idx)
                {
                    m_index -= idx;
                    for_each(m_iteratorTuple, [idx](auto &x) { x -= idx; });
                    return *this;
                }

#pragma endregion arithmeticoperators

#pragma region comparisonoperators

// if spaceship operator is available is being used we can use spaceship operator magic
#ifdef USESPACESHIP

                /**
                 * @brief Spaceship operator for comparisons
                 */
                NODISCARD ALPAKA_FN_INLINE auto operator<=>(const ZipIterator& other) const noexcept = default;

// if cpp20 *isn't* defined we get to write 70 lines of boilerplate
#else

                /**
                 * @brief Equality comparison, returns true if the iterators are the same
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator==(const ZipIterator& other) const noexcept
                {
                    return m_iteratorTuple == other.m_iteratorTuple && m_index == other.m_index;
                }

                /**
                 * @brief Inequality comparison, negated equality operator
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator!=(const ZipIterator& other) const noexcept
                {
                    return !operator==(other);
                }

                /**
                 * @brief Less than comparison, value is checked first, then index
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator<(const ZipIterator& other) const noexcept
                {
                    if(m_iteratorTuple < other.m_iteratorTuple)
                        return true;
                    if(m_iteratorTuple > other.m_iteratorTuple)
                        return false;
                    return m_index < other.m_index;
                }

                /**
                 * @brief Greater than comparison, value is checked first, then index
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator>(const ZipIterator& other) const noexcept
                {
                    if(m_iteratorTuple > other.m_iteratorTuple)
                        return true;
                    if(m_iteratorTuple < other.m_iteratorTuple)
                        return false;
                    return m_index > other.m_index;
                }

                /**
                 * @brief Less than or equal comparison, value is checked first, then index
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator<=(const ZipIterator& other) const noexcept
                {
                    if(m_iteratorTuple < other.m_iteratorTuple)
                        return true;
                    if(m_iteratorTuple > other.m_iteratorTuple)
                        return false;
                    return m_index <= other.m_index;
                }

                /**
                 * @brief Greater than or equal comparison, value is checked first, then index
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator>=(const ZipIterator& other) const noexcept
                {
                    if(m_iteratorTuple > other.m_iteratorTuple)
                        return true;
                    if(m_iteratorTuple < other.m_iteratorTuple)
                        return false;
                    return m_index >= other.m_index;
                }
#endif

#pragma endregion comparisonoperators

            private:
                IteratorTuple m_iteratorTuple;
                IdxType m_index;

                template<std::size_t I = 0, typename FuncT, typename... Tp>
                inline typename std::enable_if<I == sizeof...(Tp), void>::type
                    for_each(std::tuple<Tp...> &, FuncT) // Unused arguments are given no names
                    {
                    }

                template<std::size_t I = 0, typename FuncT, typename... Tp>
                inline typename std::enable_if<I < sizeof...(Tp), void>::type
                    for_each(std::tuple<Tp...>& t, FuncT f)
                    {
                        f(std::get<I>(t));
                        for_each<I + 1, FuncT, Tp...>(t, f);
                    }
            };

        } // namespace iterator
    } // namespace mem
} // namespace vikunja
