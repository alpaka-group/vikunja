/* Copyright 2021 Anton Reinhard
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if __cplusplus >= 202002L
#    define CPP20
#endif

#include <alpaka/alpaka.hpp>

#include <iterator>

#ifdef CPP20
#    include <compare>
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

                ConstantIterator(const DataType& value, const IdxType& idx = static_cast<IdxType>(0))
                    : v(value)
                    , index(idx)
                {
                }

                /**
                 * @brief Dereference operator to receive the stored value
                 */
                ALPAKA_FN_INLINE const DataType& operator*() const
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
                ALPAKA_FN_INLINE ConstantIterator operator+(const IdxType idx) const
                {
                    return ConstantIterator(v, index + idx);
                }

                /**
                 * @brief Add a second constant iterator of the same value to this one
                 */
                ALPAKA_FN_INLINE ConstantIterator operator+(const ConstantIterator& other) const
                {
                    assert(v == other.v && "Can't add constant iterators of different values!");
                    return ConstantIterator(v, index + other.index);
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                ALPAKA_FN_INLINE ConstantIterator operator-(const IdxType idx) const
                {
                    return ConstantIterator(v, index - idx);
                }

                /**
                 * @brief Subtract a second constant iterator of the same value from this one
                 */
                ALPAKA_FN_INLINE ConstantIterator operator-(const ConstantIterator& other) const
                {
                    assert(v == other.v && "Can't subtract constant iterators of different values!");
                    return ConstantIterator(v, index - other.index);
                }

                /**
                 * @brief Add an index to this iterator
                 */
                ALPAKA_FN_INLINE ConstantIterator& operator+=(const IdxType idx)
                {
                    index += idx;
                    return *this;
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                ALPAKA_FN_INLINE ConstantIterator& operator-=(const IdxType idx)
                {
                    index -= idx;
                    return *this;
                }
#pragma endregion arithmeticoperators

#pragma region comparisonoperators

// if cpp20 is being used we can use starship operator magic
#ifdef CPP20

                /**
                 * @brief Starship operator for comparisons
                 */
                auto operator<=>(const ConstantIterator& other) const noexcept = default;

// if cpp20 *isn't* defined we get to write 70 lines of boilerplate
#else

                /**
                 * @brief Equality comparison, returns true if the iterators are the same
                 */
                bool operator==(const ConstantIterator& other) const noexcept
                {
                    return v == other.v && index == other.index;
                }

                /**
                 * @brief Inequality comparison, negated equality operator
                 */
                bool operator!=(const ConstantIterator& other) const noexcept
                {
                    return !operator==(other);
                }

                /**
                 * @brief Less than comparison, value is checked first, then index
                 */
                bool operator<(const ConstantIterator& other) const noexcept
                {
                    if(v < other.v)
                        return true;
                    if(v > other.v)
                        return false;
                    return index < other.index;
                }

                /**
                 * @brief Greater than comparison, value is checked first, then index
                 */
                bool operator>(const ConstantIterator& other) const noexcept
                {
                    if(v > other.v)
                        return true;
                    if(v < other.v)
                        return false;
                    return index > other.index;
                }

                /**
                 * @brief Less than or equal comparison, value is checked first, then index
                 */
                bool operator<=(const ConstantIterator& other) const noexcept
                {
                    if(v < other.v)
                        return true;
                    if(v > other.v)
                        return false;
                    return index <= other.index;
                }

                /**
                 * @brief Greater than or equal comparison, value is checked first, then index
                 */
                bool operator>=(const ConstantIterator& other) const noexcept
                {
                    if(v > other.v)
                        return true;
                    if(v < other.v)
                        return false;
                    return index >= other.index;
                }
#endif

#pragma endregion comparisonoperators

            private:
                const DataType v;
                IdxType index;
            };

        } // namespace iterator
    } // namespace mem
} // namespace vikunja