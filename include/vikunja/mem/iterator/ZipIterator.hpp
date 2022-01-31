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
             * @tparam TIteratorTuplePtr The type of the data
             * @tparam TIteratorTupleVal The type of the data
             * @tparam TIdx The type of the index
             */
            template<typename TIteratorTuplePtr, typename TIteratorTupleVal, typename TIdx = int64_t>
            class ZipIterator
            {
            public:
                // Need all 5 of these types for iterator_traits
                using reference = TIteratorTupleVal&;
                using value_type = TIteratorTupleVal;
                using pointer = TIteratorTupleVal*;
                using difference_type = TIdx;
                using iterator_category = std::random_access_iterator_tag;

                /**
                 * @brief Constructor for the ZipIterator
                 * @param iteratorTuplePtr The tuple to initialize the iterator with
                 * @param idx The index for the iterator, default 0
                 */
                ALPAKA_FN_HOST_ACC ZipIterator(TIteratorTuplePtr iteratorTuplePtr, const TIdx& idx = static_cast<TIdx>(0))
                    : m_index(idx)
                    , m_iteratorTuplePtr(iteratorTuplePtr)
                    , m_iteratorTupleVal(makeValueTuple(m_iteratorTuplePtr))
                {
                    if (idx != 0)
                    {
                        forEach(m_iteratorTuplePtr, [idx](auto &x) { x += idx; });
                        m_iteratorTupleVal = makeValueTuple(m_iteratorTuplePtr);
                    }
                }

                /**
                 * @brief Dereference operator to receive the stored value
                 */
                NODISCARD ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE TIteratorTupleVal& operator*()
                {
                    return m_iteratorTupleVal;
                }

                /**
                 * @brief Index operator to get stored value at some given offset from this iterator
                 */
                NODISCARD ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE TIteratorTupleVal operator[](const TIdx idx) const
                {
                    TIdx indexDiff = idx - m_index;
                    return (*this + indexDiff).operator*();
                }

                // NODISCARD ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE TIteratorTupleVal& operator=(TIteratorTupleVal iteratorTupleVal)
                // {
                //     updateIteratorTupleValue(iteratorTupleVal);
                //     m_iteratorTupleVal = makeValueTuple(m_iteratorTuplePtr);
                //     return m_iteratorTupleVal;
                // }

#pragma region arithmeticoperators
                /**
                 * @brief Prefix increment operator
                 */
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE ZipIterator& operator++()
                {
                    ++m_index;
                    forEach(m_iteratorTuplePtr, [](auto &x) { ++x; });
                    m_iteratorTupleVal = makeValueTuple(m_iteratorTuplePtr);
                    return *this;
                }

                /**
                 * @brief Postfix increment operator
                 * @note Use prefix increment operator instead if possible to avoid copies
                 */
                ALPAKA_FN_HOST_ACC ZipIterator operator++(int)
                {
                    ZipIterator tmp = *this;
                    ++m_index;
                    forEach(m_iteratorTuplePtr, [](auto &x) { ++x; });
                    m_iteratorTupleVal = makeValueTuple(m_iteratorTuplePtr);
                    return tmp;
                }

                /**
                 * @brief Prefix decrement operator
                 */
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE ZipIterator& operator--()
                {
                    --m_index;
                    forEach(m_iteratorTuplePtr, [](auto &x) { --x; });
                    m_iteratorTupleVal = makeValueTuple(m_iteratorTuplePtr);
                    return *this;
                }

                /**
                 * @brief Postfix decrement operator
                 * @note Use prefix decrement operator instead if possible to avoid copies
                 */
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE ZipIterator operator--(int)
                {
                    ZipIterator tmp = *this;
                    --m_index;
                    forEach(m_iteratorTuplePtr, [](auto &x) { --x; });
                    m_iteratorTupleVal = makeValueTuple(m_iteratorTuplePtr);
                    return tmp;
                }

                /**
                 * @brief Add an index to this iterator
                 */
                NODISCARD ALPAKA_FN_HOST_ACC friend ALPAKA_FN_INLINE ZipIterator operator+(ZipIterator zipIter, const TIdx idx)
                {
                    zipIter += idx;
                    return zipIter;
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                NODISCARD ALPAKA_FN_HOST_ACC friend ALPAKA_FN_INLINE ZipIterator operator-(ZipIterator zipIter, const TIdx idx)
                {
                    zipIter -= idx;
                    return zipIter;
                }

                NODISCARD ALPAKA_FN_HOST_ACC friend ALPAKA_FN_INLINE TIdx operator-(const ZipIterator& zipIter, const ZipIterator& other)
                {
                    return zipIter.m_index - other.m_index;
                }

                /**
                 * @brief Add an index to this iterator
                 */
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE ZipIterator& operator+=(const TIdx idx)
                {
                    m_index += idx;
                    forEach(m_iteratorTuplePtr, [idx](auto &x) { x += idx; });
                    m_iteratorTupleVal = makeValueTuple(m_iteratorTuplePtr);
                    return *this;
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE ZipIterator& operator-=(const TIdx idx)
                {
                    m_index -= idx;
                    forEach(m_iteratorTuplePtr, [idx](auto &x) { x -= idx; });
                    m_iteratorTupleVal = makeValueTuple(m_iteratorTuplePtr);
                    return *this;
                }

#pragma endregion arithmeticoperators

#pragma region comparisonoperators

#ifdef USESPACESHIP

                /**
                 * @brief Spaceship operator for comparisons
                 */
                NODISCARD ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator<=>(const ZipIterator& other) const noexcept
                {
                    return m_iteratorTuplePtr.operator<=>(other.m_iteratorTuplePtr);
                }

#else

                /**
                 * @brief Equality comparison, returns true if the index are the same
                 */
                NODISCARD ALPAKA_FN_HOST_ACC friend ALPAKA_FN_INLINE bool operator==(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.m_index == other.m_index;
                }

                /**
                 * @brief Inequality comparison, negated equality operator
                 */
                NODISCARD ALPAKA_FN_HOST_ACC friend ALPAKA_FN_INLINE bool operator!=(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return !operator==(zipIter, other);
                }

                /**
                 * @brief Less than comparison, index is checked 
                 */
                NODISCARD ALPAKA_FN_HOST_ACC friend ALPAKA_FN_INLINE bool operator<(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.m_index < other.m_index;
                }

                /**
                 * @brief Greater than comparison, index is checked 
                 */
                NODISCARD ALPAKA_FN_HOST_ACC friend ALPAKA_FN_INLINE bool operator>(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.m_index > other.m_index;
                }

                /**
                 * @brief Less than or equal comparison, index is checked 
                 */
                NODISCARD ALPAKA_FN_HOST_ACC friend ALPAKA_FN_INLINE bool operator<=(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.m_index <= other.m_index;
                }

                /**
                 * @brief Greater than or equal comparison, index is checked 
                 */
                NODISCARD ALPAKA_FN_HOST_ACC friend ALPAKA_FN_INLINE bool operator>=(const ZipIterator& zipIter, const ZipIterator& other) noexcept
                {
                    return zipIter.m_index >= other.m_index;
                }
#endif

#pragma endregion comparisonoperators

            private:
                TIdx m_index;
                TIteratorTuplePtr m_iteratorTuplePtr;
                TIteratorTupleVal m_iteratorTupleVal;

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

                // template<std::size_t I = 0, typename... Tp>
                // inline typename std::enable_if<I == sizeof...(Tp), void>::type updateIteratorTupleValue(std::tuple<Tp...> &) // Unused arguments are given no names
                // {
                // }

                // template<std::size_t I = 0, typename... Tp>
                // inline typename std::enable_if<I < sizeof...(Tp), void>::type updateIteratorTupleValue(std::tuple<Tp...>& t)
                // {
                //     *std::get<I>(m_iteratorTuplePtr) = std::get<I>(t);
                //     updateIteratorTupleValue<I + 1, Tp...>(t);
                // }
            };

        } // namespace iterator
    } // namespace mem
} // namespace vikunja
