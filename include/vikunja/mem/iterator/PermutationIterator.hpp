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
            template<typename ElementIterator, typename IndexIterator, typename IdxType = int64_t>
            class PermutationIterator
            {
            public:
                using difference_type = int64_t;
                using value_type = typename std::iterator_traits<ElementIterator>::value_type;
                using pointer = typename std::iterator_traits<ElementIterator>::pointer;
                using reference = typename std::iterator_traits<ElementIterator>::reference;
                using iterator_category = std::random_access_iterator_tag;

                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! \param data Data.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
                PermutationIterator(const ElementIterator& _elIter, const IndexIterator& _idxIter, const IdxType& _idx = static_cast<IdxType>(0))
                    : elIter(_elIter)
                    , idxIter(_idxIter)
                    , idx(_idx)
                {
                }
                
                /**
                 * @brief Dereference operator to receive the stored value
                 */
                NODISCARD ALPAKA_FN_INLINE reference operator*() const
                {
                    return *(elIter + *idxIter);
                }

                /**
                 * @brief Index operator to get stored value at some given offset from this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE reference operator[](int _idx)
                {
                    static_assert(std::is_convertible<std::random_access_iterator_tag, typename std::iterator_traits<ElementIterator>::iterator_category>::value, 
                        "The function only accepts random access iterators or raw pointers to an array.\n");
                    std::advance(idxIter, _idx - idx);
                    idx = _idx;
                    return elIter[*(idxIter)];
                }

                /**
                 * @brief Index operator to get stored value at some given offset from this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator begin()
                {
                    return PermutationIterator(elIter, idxIter, 0);
                }

                /**
                 * @brief Index operator to get stored value at some given offset from this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator end()
                {
                    return PermutationIterator(elIter, idxIter, 3);
                }

#pragma region arithmeticoperators
                /**
                 * @brief Prefix increment operator
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator& operator++()
                {
                    ++idx;
                    return *this;
                }

                /**
                 * @brief Postfix increment operator
                 * @note Use prefix increment operator instead if possible to avoid copies
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator operator++(int)
                {
                    PermutationIterator cpy = *this;
                    ++idx;
                    return cpy;
                }

                /**
                 * @brief Prefix decrement operator
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator& operator--()
                {
                    // static_assert(std::is_convertible<std::bidirectional_iterator_tag, typename std::iterator_traits<IndexIterator>::iterator_category>::value, 
                    //     "The function only accepts random access iterators or raw pointers to an array.\n");
                    --idx;
                    return *this;
                }

                /**
                 * @brief Postfix decrement operator
                 * @note Use prefix decrement operator instead if possible to avoid copies
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator operator--(int)
                {
                    // static_assert(std::is_convertible<std::bidirectional_iterator_tag, typename std::iterator_traits<IndexIterator>::iterator_category>::value, 
                    //     "The function only accepts random access iterators or raw pointers to an array.\n");
                    PermutationIterator cpy = *this;
                    --idx;
                    return cpy;
                }

                /**
                 * @brief Add an index to this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator operator+(const IdxType _idx) const
                {
                    return PermutationIterator(elIter, idxIter, idx + _idx);
                }

                /**
                 * @brief Add a second constant iterator of the same value to this one
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator operator+(const PermutationIterator& other) const
                {
                    return PermutationIterator(elIter.insert(std::end(elIter), std::begin(other.elIter), std::end(other.elIter)), idxIter.insert(std::end(idxIter), std::begin(other.idxIter), std::end(other.idxIter)));
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator operator-(const IdxType _idx) const
                {
                    return PermutationIterator(elIter, idxIter);
                }

                /**
                 * @brief Subtract a second constant iterator of the same value from this one
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator operator-(const PermutationIterator& other) const
                {
                    return PermutationIterator(elIter, idxIter);
                }

                /**
                 * @brief Add an index to this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator& operator+=(const IdxType _idx)
                {
                    idx += _idx;
                    return *this;
                }

                /**
                 * @brief Subtract an index from this iterator
                 */
                NODISCARD ALPAKA_FN_INLINE PermutationIterator& operator-=(const IdxType _idx)
                {
                    idx -= _idx;
                    return *this;
                }

#pragma endregion arithmeticoperators

#pragma region comparisonoperators

// if spaceship operator is available is being used we can use spaceship operator magic
#ifdef USESPACESHIP

                /**
                 * @brief Spaceship operator for comparisons
                 */
                NODISCARD ALPAKA_FN_INLINE auto operator<=>(const PermutationIterator& other) const noexcept = default;

// if cpp20 *isn't* defined we get to write 70 lines of boilerplate
#else

                /**
                 * @brief Equality comparison, returns true if the iterators are the same
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator==(const PermutationIterator& other) const noexcept
                {
                    return elIter == other.elIter && idxIter == other.idxIter;
                }

                /**
                 * @brief Inequality comparison, negated equality operator
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator!=(const PermutationIterator& other) const noexcept
                {
                    return !operator==(other);
                }

                /**
                 * @brief Less than comparison, value is checked first, then index
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator<(const PermutationIterator& other) const noexcept
                {
                    if (elIter < other.elIter)
                        return true;
                    if (elIter > other.elIter)
                        return false;
                    return idxIter < other.idxIter;
                }

                /**
                 * @brief Greater than comparison, value is checked first, then index
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator>(const PermutationIterator& other) const noexcept
                {
                    if (elIter > other.elIter)
                        return true;
                    if (elIter < other.elIter)
                        return false;
                    return idxIter > other.idxIter;
                }

                /**
                 * @brief Less than or equal comparison, value is checked first, then index
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator<=(const PermutationIterator& other) const noexcept
                {
                    if (elIter < other.elIter)
                        return true;
                    if (elIter > other.elIter)
                        return false;
                    return idxIter <= other.idxIter;
                }

                /**
                 * @brief Greater than or equal comparison, value is checked first, then index
                 */
                NODISCARD ALPAKA_FN_INLINE bool operator>=(const PermutationIterator& other) const noexcept
                {
                    if (elIter > other.elIter)
                        return true;
                    if (elIter < other.elIter)
                        return false;
                    return idxIter >= other.idxIter;
                }
#endif

#pragma endregion comparisonoperators

            private:
                ElementIterator elIter;
                IndexIterator idxIter;
                IdxType idx;
            };
        } // namespace iterator
    } // namespace mem
} // namespace vikunja
