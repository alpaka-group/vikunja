/* Copyright 2022 Anton Reinhard
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <iterator>
#include <memory>

#if __has_include(<compare>)
#    include <compare>
#    if defined(__cpp_impl_three_way_comparison) && defined(__cpp_lib_three_way_comparison)
#        define USESPACESHIP
#    endif
#endif

// TODO: figure out why it doesn't work
#undef USESPACESHIP

#if __has_cpp_attribute(nodiscard)
#    define NODISCARD [[nodiscard]]
#else
#    define NODISCARD
#endif

namespace vikunja::mem::iterator
{
    /**
     * @brief A transform iterator, returning a value given initially at any index. As such it has no bounds,
     * other than the bounds of the index type used.
     * @tparam DataType The type of the data
     * @tparam TInputIterator The type of the underlying input iterator
     * @tparam TTransform The Functor transforming the input values
     */
    template<typename AccObj, typename DataType, typename TInputIterator, typename TTransform>
    class TransformIterator
    {
    public:
        using IdxType = int64_t;

        // Need all 5 of these types for iterator_traits
        using difference_type = IdxType;
        using value_type = DataType;
        using pointer = DataType*;
        using reference = DataType&;
        using iterator_category = std::random_access_iterator_tag;

        using TInputIteratorDeref = decltype(*std::declval<TInputIterator>());
        using AccObjPtr = AccObj*;

        /**
         * @brief Constructor for the TransformIterator
         * @param it The underlying input iterator that this iterator will walk on and transform
         * @param func The function applied to the input iterator's values
         */
        constexpr TransformIterator(const TInputIterator& it, const TTransform func)
            : it(it)
            , func(func)
            , accelerator(nullptr)
        {
        }

        /**
         * @brief Set accelerator object
         */
        void setAcc(AccObjPtr acc)
        {
            accelerator = std::move(acc);
        }

        /**
         * @brief Dereference operator to receive the stored value, invoking without accelerator
         */
        template<typename Trans = TTransform, typename Acc = AccObjPtr, typename It = TInputIteratorDeref>
        NODISCARD constexpr ALPAKA_FN_INLINE std::enable_if_t<
            std::negation_v<std::is_invocable<
                Trans,
                Acc,
                It>> && std::is_same_v<Trans, TTransform> && std::is_same_v<Acc, AccObjPtr> && std::is_same_v<It, TInputIteratorDeref>,
            DataType>
        operator*() const
        {
            return func(*it);
        }

        /**
         * @brief Dereference operator to receive the stored value, invoking with accelerator
         */
        template<typename Trans = TTransform, typename Acc = AccObjPtr, typename It = TInputIteratorDeref>
        NODISCARD constexpr ALPAKA_FN_INLINE std::enable_if_t<
            std::is_invocable_v<
                TTransform,
                AccObjPtr,
                TInputIteratorDeref> && std::is_same_v<Trans, TTransform> && std::is_same_v<Acc, AccObjPtr> && std::is_same_v<It, TInputIteratorDeref>,
            DataType>
        operator*() const
        {
            static_assert(accelerator != nullptr);
            return func(accelerator, *it);
        }

        /**
         * @brief Index operator to get stored value at some given offset from this iterator
         */
        NODISCARD constexpr ALPAKA_FN_INLINE DataType operator[](const IdxType idx) const
        {
            return (*this + idx).operator*();
        }

#pragma region arithmeticoperators
        /**
         * @brief Postfix increment operator
         * @note Use prefix increment operator instead if possible to avoid copies
         */
        constexpr ALPAKA_FN_INLINE TransformIterator operator++()
        {
            TransformIterator cpy = *this;
            ++it;
            return cpy;
        }

        /**
         * @brief Prefix increment operator
         */
        constexpr ALPAKA_FN_INLINE TransformIterator& operator++(int)
        {
            ++it;
            return *this;
        }

        /**
         * @brief Postfix decrement operator
         * @note Use prefix decrement operator instead if possible to avoid copies
         */
        constexpr ALPAKA_FN_INLINE TransformIterator operator--()
        {
            TransformIterator cpy = *this;
            --it;
            return cpy;
        }

        /**
         * @brief Prefix decrement operator
         */
        constexpr ALPAKA_FN_INLINE TransformIterator& operator--(int)
        {
            --it;
            return *this;
        }

        /**
         * @brief Add an index to this iterator
         */
        NODISCARD constexpr friend ALPAKA_FN_INLINE TransformIterator
        operator+(TransformIterator it, const IdxType idx)
        {
            it.it += idx;
            return it;
        }

        /**
         * @brief Subtract an index from this iterator
         */
        NODISCARD constexpr friend ALPAKA_FN_INLINE TransformIterator
        operator-(TransformIterator it, const IdxType idx)
        {
            it.it -= idx;
            return it;
        }

        /**
         * @brief Subtract a second transform iterator of the same value from this one
         */
        NODISCARD constexpr friend ALPAKA_FN_INLINE IdxType
        operator-(TransformIterator it, const TransformIterator& other)
        {
            return it.it - other.it;
        }

        /**
         * @brief Add an index to this iterator
         */
        constexpr friend ALPAKA_FN_INLINE TransformIterator& operator+=(TransformIterator& it, const IdxType idx)
        {
            it.it += idx;
            return it;
        }

        /**
         * @brief Subtract an index from this iterator
         */
        constexpr friend ALPAKA_FN_INLINE TransformIterator& operator-=(TransformIterator& it, const IdxType idx)
        {
            it.it -= idx;
            return it;
        }

#pragma endregion arithmeticoperators

#pragma region comparisonoperators

#ifdef USESPACESHIP

        NODISCARD constexpr ALPAKA_FN_INLINE std::strong_ordering operator<=>(
            const TransformIterator& other) const noexcept
        {
            return it.operator<=>(other.it);
        }

#else

        NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator==(
            const TransformIterator& it,
            const TransformIterator& other) noexcept
        {
            return it.it == other.it;
        }

        NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator!=(
            const TransformIterator& it,
            const TransformIterator& other) noexcept
        {
            return !operator==(it, other);
        }

        NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator<(
            const TransformIterator& it,
            const TransformIterator& other) noexcept
        {
            return it.it < other.it;
        }

        NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator>(
            const TransformIterator& it,
            const TransformIterator& other) noexcept
        {
            return operator<(other, it);
        }

        NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator<=(
            const TransformIterator& it,
            const TransformIterator& other) noexcept
        {
            return operator<(it, other) || operator==(it, other);
        }

        NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator>=(
            const TransformIterator& it,
            const TransformIterator& other) noexcept
        {
            return operator<=(other, it);
        }
#endif

#pragma endregion comparisonoperators

    private:
        TInputIterator it;
        TTransform func;

        AccObjPtr accelerator;
    };

} // namespace vikunja::mem::iterator
