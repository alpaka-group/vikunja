/* Copyright 2021 Anton Reinhard
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

#if defined(__cpp_impl_three_way_comparison)
#    define USESPACESHIP
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
             * @brief A constant iterator, returning a value given initially at any index. As such it has no bounds,
             * other than the bounds of the index type used.
             * @tparam DataType The type of the data
             * @tparam IdxType The type of the index
             */
            template<typename DataType, typename IdxType = int64_t>
            class ConstantIterator
            {
            public:
                // Need all 5 of these types for iterator_traits
                using difference_type = IdxType;
                using value_type = DataType;
                using pointer = DataType*;
                using reference = DataType&;
                using iterator_category = std::random_access_iterator_tag;

                /**
                 * @brief Constructor for the ConstantIterator
                 * @param value The value to initialize the iterator with
                 * @param idx The index for the iterator, default 0
                 */
                ConstantIterator(const DataType& value, const IdxType& idx = {}) : v(value), index(idx)
                {
                }

                /**
                 * @brief Dereference operator to receive the stored value
                 */
                NODISCARD ALPAKA_FN_INLINE const DataType& operator*() const
                {
                    return v;
                }

                /**
                 * @brief Index operator to get stored value at some given offset from this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE const DataType& operator[](int) const
                {
                    return v;
                }

#pragma region arithmeticoperators
                /**
                 * @brief Postfix increment operator
                 * @note Use prefix increment operator instead if possible to avoid copies
                 */
                ALPAKA_FN_INLINE ConstantIterator operator++()
                {
                    ConstantIterator cpy = *this;
                    ++index;
                    return cpy;
                }

                /**
                 * @brief Prefix increment operator
                 */
                ALPAKA_FN_INLINE ConstantIterator& operator++(int)
                {
                    ++index;
                    return *this;
                }

                /**
                 * @brief Postfix decrement operator
                 * @note Use prefix decrement operator instead if possible to avoid copies
                 */
                ALPAKA_FN_INLINE ConstantIterator operator--()
                {
                    ConstantIterator cpy = *this;
                    --index;
                    return cpy;
                }

                /**
                 * @brief Prefix decrement operator
                 */
                ALPAKA_FN_INLINE ConstantIterator operator--(int)
                {
                    --index;
                    return *this;
                }

                /**
                 * @brief Add an index to this iterator
                 */
                NODISCARD friend ALPAKA_FN_INLINE ConstantIterator operator+(ConstantIterator it, IdxType idx)
                {
                    return it += idx;
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                NODISCARD friend ALPAKA_FN_INLINE ConstantIterator operator-(ConstantIterator it, const IdxType idx)
                {
                    return it -= idx;
                }

                /**
                 * @brief Subtract a second constant iterator of the same value from this one
                 */
                NODISCARD friend ALPAKA_FN_INLINE IdxType operator-(ConstantIterator it, const ConstantIterator& other)
                {
                    assert(it.v == other.v && "Can't subtract constant iterators of different values!");
                    return it.index - other.index;
                }

                /**
                 * @brief Add an index to this iterator
                 */
                friend ALPAKA_FN_INLINE ConstantIterator& operator+=(ConstantIterator& it, const IdxType idx)
                {
                    it.index += idx;
                    return it;
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                friend ALPAKA_FN_INLINE ConstantIterator& operator-=(ConstantIterator& it, const IdxType idx)
                {
                    it.index -= idx;
                    return it;
                }

#pragma endregion arithmeticoperators

#pragma region comparisonoperators

#ifdef USESPACESHIP

                NODISCARD ALPAKA_FN_INLINE auto operator<=>(const ConstantIterator& other) const noexcept = default;

#else

                NODISCARD friend ALPAKA_FN_INLINE bool operator==(
                    const ConstantIterator& it,
                    const ConstantIterator& other) noexcept
                {
                    return it.v == other.v && it.index == other.index;
                }

                NODISCARD friend ALPAKA_FN_INLINE bool operator!=(
                    const ConstantIterator& it,
                    const ConstantIterator& other) noexcept
                {
                    return !operator==(it, other);
                }

                NODISCARD friend ALPAKA_FN_INLINE bool operator<(
                    const ConstantIterator& it,
                    const ConstantIterator& other) noexcept
                {
                    if(it.v < other.v)
                        return true;
                    if(it.v > other.v)
                        return false;
                    return it.index < other.index;
                }

                NODISCARD friend ALPAKA_FN_INLINE bool operator>(
                    const ConstantIterator& it,
                    const ConstantIterator& other) noexcept
                {
                    return operator<(other, it);
                }

                NODISCARD friend ALPAKA_FN_INLINE bool operator<=(
                    const ConstantIterator& it,
                    const ConstantIterator& other) noexcept
                {
                    return operator<(it, other) || operator==(it, other);
                }

                NODISCARD friend ALPAKA_FN_INLINE bool operator>=(
                    const ConstantIterator& it,
                    const ConstantIterator& other) noexcept
                {
                    return operator<=(other, it);
                }
#endif

#pragma endregion comparisonoperators

            private:
                DataType v;
                IdxType index;
            };

        } // namespace iterator
    } // namespace mem
} // namespace vikunja
