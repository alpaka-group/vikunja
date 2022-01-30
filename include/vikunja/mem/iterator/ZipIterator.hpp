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
             * @tparam IteratorTuplePtr The type of the data
             * @tparam IteratorTupleVal The type of the data
             * @tparam IdxType The type of the index
             */
            template<typename IteratorTuplePtr, typename IteratorTupleVal, typename IdxType = int64_t>
            class ZipIterator
            {
            public:
                // Need all 5 of these types for iterator_traits
                using reference = IteratorTupleVal&;
                using value_type = IteratorTupleVal;
                using pointer = IteratorTupleVal*;
                using difference_type = IdxType;
                using iterator_category = std::random_access_iterator_tag;

                /**
                 * @brief Constructor for the ZipIterator
                 * @param iteratorTuplePtr The tuple to initialize the iterator with
                 * @param idx The index for the iterator, default 0
                 */
                constexpr ZipIterator(IteratorTuplePtr iteratorTuplePtr, const IdxType& idx = static_cast<IdxType>(0))
                    : mIndex(idx)
                    , mIteratorTuplePtr(iteratorTuplePtr)
                    , mIteratorTupleVal(makeValueTuple(mIteratorTuplePtr))
                {
                    if (idx != 0)
                    {
                        forEach(mIteratorTuplePtr, [idx](auto &x) { x += idx; });
                        mIteratorTupleVal = makeValueTuple(mIteratorTuplePtr);
                    }
                }

                /**
                 * @brief Dereference operator to receive the stored value
                 */
                NODISCARD constexpr ALPAKA_FN_INLINE IteratorTupleVal& operator*()
                {
                    return mIteratorTupleVal;
                }

                /**
                 * @brief Index operator to get stored value at some given offset from this iterator
                 */
                NODISCARD constexpr ALPAKA_FN_INLINE const IteratorTupleVal operator[](const IdxType idx)
                {
                    IteratorTuplePtr tmp = mIteratorTuplePtr;
                    IdxType indexDiff = idx - mIndex;
                    forEach(tmp, [indexDiff](auto &x) { x += indexDiff; });
                    return makeValueTuple(tmp);
                }

                // NODISCARD constexpr ALPAKA_FN_INLINE IteratorTupleVal& operator=(IteratorTupleVal iteratorTupleVal)
                // {
                //     updateIteratorTupleValue(iteratorTupleVal);
                //     mIteratorTupleVal = makeValueTuple(mIteratorTuplePtr);
                //     return mIteratorTupleVal;
                // }

#pragma region arithmeticoperators
                /**
                 * @brief Prefix increment operator
                 */
                constexpr ALPAKA_FN_INLINE ZipIterator& operator++()
                {
                    ++mIndex;
                    forEach(mIteratorTuplePtr, [](auto &x) { ++x; });
                    mIteratorTupleVal = makeValueTuple(mIteratorTuplePtr);
                    return *this;
                }

                /**
                 * @brief Postfix increment operator
                 * @note Use prefix increment operator instead if possible to avoid copies
                 */
                constexpr ZipIterator operator++(int)
                {
                    ZipIterator tmp = *this;
                    ++mIndex;
                    forEach(mIteratorTuplePtr, [](auto &x) { ++x; });
                    mIteratorTupleVal = makeValueTuple(mIteratorTuplePtr);
                    return tmp;
                }

                /**
                 * @brief Prefix decrement operator
                 */
                constexpr ALPAKA_FN_INLINE ZipIterator& operator--()
                {
                    --mIndex;
                    forEach(mIteratorTuplePtr, [](auto &x) { --x; });
                    mIteratorTupleVal = makeValueTuple(mIteratorTuplePtr);
                    return *this;
                }

                /**
                 * @brief Postfix decrement operator
                 * @note Use prefix decrement operator instead if possible to avoid copies
                 */
                constexpr ALPAKA_FN_INLINE ZipIterator operator--(int)
                {
                    ZipIterator tmp = *this;
                    --mIndex;
                    forEach(mIteratorTuplePtr, [](auto &x) { --x; });
                    mIteratorTupleVal = makeValueTuple(mIteratorTuplePtr);
                    return tmp;
                }

                /**
                 * @brief Add an index to this iterator
                 */
                NODISCARD constexpr friend ALPAKA_FN_INLINE ZipIterator operator+(ZipIterator zipIter, const IdxType idx)
                {
                    zipIter += idx;
                    return zipIter;
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                NODISCARD constexpr friend ALPAKA_FN_INLINE ZipIterator operator-(ZipIterator zipIter, const IdxType idx)
                {
                    zipIter -= idx;
                    return zipIter;
                }

                /**
                 * @brief Add an index to this iterator
                 */
                constexpr ALPAKA_FN_INLINE ZipIterator& operator+=(const IdxType idx)
                {
                    mIndex += idx;
                    forEach(mIteratorTuplePtr, [idx](auto &x) { x += idx; });
                    mIteratorTupleVal = makeValueTuple(mIteratorTuplePtr);
                    return *this;
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                constexpr ALPAKA_FN_INLINE ZipIterator& operator-=(const IdxType idx)
                {
                    mIndex -= idx;
                    forEach(mIteratorTuplePtr, [idx](auto &x) { x -= idx; });
                    mIteratorTupleVal = makeValueTuple(mIteratorTuplePtr);
                    return *this;
                }

#pragma endregion arithmeticoperators

#pragma region comparisonoperators

#ifdef USESPACESHIP

                /**
                 * @brief Spaceship operator for comparisons
                 */
                NODISCARD constexpr ALPAKA_FN_INLINE auto operator<=>(const ZipIterator& other) const noexcept
                {
                    return mIteratorTuplePtr.operator<=>(other.mIteratorTuplePtr);
                }

#else

                /**
                 * @brief Equality comparison, returns true if the index are the same
                 */
                NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator==(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.mIndex == other.mIndex;
                }

                /**
                 * @brief Inequality comparison, negated equality operator
                 */
                NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator!=(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return !operator==(zipIter, other);
                }

                /**
                 * @brief Less than comparison, index is checked 
                 */
                NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator<(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.mIndex < other.mIndex;
                }

                /**
                 * @brief Greater than comparison, index is checked 
                 */
                NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator>(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.mIndex > other.mIndex;
                }

                /**
                 * @brief Less than or equal comparison, index is checked 
                 */
                NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator<=(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.mIndex <= other.mIndex;
                }

                /**
                 * @brief Greater than or equal comparison, index is checked 
                 */
                NODISCARD constexpr friend ALPAKA_FN_INLINE bool operator>=(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.mIndex >= other.mIndex;
                }
#endif

#pragma endregion comparisonoperators

            private:
                IdxType mIndex;
                IteratorTuplePtr mIteratorTuplePtr;
                IteratorTupleVal mIteratorTupleVal;

                template<int... Is>
                struct seq { };
            
                template<int N, int... Is>
                struct gen_seq : gen_seq<N - 1, N - 1, Is...> { };
            
                template<int... Is>
                struct gen_seq<0, Is...> : seq<Is...> { };

                template<typename... Args, int... Is>
                auto makeValueTuple(std::tuple<Args...>& t, seq<Is...>)
                    -> std::tuple<typename std::remove_pointer<Args>::type...>
                {
                    return std::forward_as_tuple(*std::get<Is>(t)...);
                }
            
                template<typename... Args>
                auto makeValueTuple(std::tuple<Args...>& t)
                    -> std::tuple<typename std::remove_pointer<Args>::type...>
                {
                    return makeValueTuple(t, gen_seq<sizeof...(Args)>());
                }

                template<std::size_t I = 0, typename FuncT, typename... Tp>
                inline typename std::enable_if<I == sizeof...(Tp), void>::type forEach(std::tuple<Tp...> &, FuncT) // Unused arguments are given no names
                {
                }

                template<std::size_t I = 0, typename FuncT, typename... Tp>
                inline typename std::enable_if<I < sizeof...(Tp), void>::type forEach(std::tuple<Tp...>& t, FuncT f)
                {
                    f(std::get<I>(t));
                    forEach<I + 1, FuncT, Tp...>(t, f);
                }

                template<std::size_t I = 0, typename... Tp>
                inline typename std::enable_if<I == sizeof...(Tp), void>::type updateIteratorTupleValue(std::tuple<Tp...> &) // Unused arguments are given no names
                {
                }

                template<std::size_t I = 0, typename... Tp>
                inline typename std::enable_if<I < sizeof...(Tp), void>::type updateIteratorTupleValue(std::tuple<Tp...>& t)
                {
                    *std::get<I>(mIteratorTuplePtr) = std::get<I>(t);
                    updateIteratorTupleValue<I + 1, Tp...>(t);
                }
            };

        } // namespace iterator
    } // namespace mem
} // namespace vikunja
