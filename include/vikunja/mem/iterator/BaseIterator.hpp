//
// Created by mewes30 on 16.01.19.
//

#pragma once

#include <alpaka/alpaka.hpp>

namespace vikunja {
    namespace mem {
        namespace iterator {

            // TODO: this class is from Jonas Schenke

            //! An iterator base class.
            //!
            //! \tparam T The type.
            //! \tparam TBuf The buffer type (standard is T).
            template <typename T, typename TInputIterator = T*>
            class BaseIterator
            {
            protected:
                TInputIterator const mData; // The underlying iterator should not be changed
                uint64_t mIndex;
                const uint64_t mMaximum;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! \param data A pointer to the data.
                //! \param index The index.
                //! \param maximum The first index outside of the iterator memory.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE BaseIterator(TInputIterator const &data,
                                                             uint64_t index,
                                                             uint64_t maximum)
                        : mData(data), mIndex(index), mMaximum(maximum)
                {
                }

                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! \param other The other iterator object.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE BaseIterator(const BaseIterator &other) = default;

                //-----------------------------------------------------------------------------
                //! Compare operator.
                //!
                //! \param other The other object.
                //!
                //! Returns true if objects are equal and false otherwise.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
                operator==(const BaseIterator &other) const -> bool
                {
                    return (this->mData == other.mData) && (this->mIndex == other.mIndex) &&
                           (this->mMaximum == other.mMaximum);
                }

                //-----------------------------------------------------------------------------
                //! Compare operator.
                //!
                //! \param other The other object.
                //!
                //! Returns false if objects are equal and true otherwise.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
                operator!=(const BaseIterator &other) const -> bool
                {
                    return !operator==(other);
                }

                //-----------------------------------------------------------------------------
                //! Compare operator.
                //!
                //! \param other The other object.
                //!
                //! Returns false if the other object is equal or smaller and true
                //! otherwise.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
                operator<(const BaseIterator &other) const -> bool
                {
                    return mIndex < other.mIndex;
                }

                //-----------------------------------------------------------------------------
                //! Compare operator.
                //!
                //! \param other The other object.
                //!
                //! Returns false if the other object is equal or bigger and true otherwise.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
                operator>(const BaseIterator &other) const -> bool
                {
                    return mIndex > other.mIndex;
                }

                //-----------------------------------------------------------------------------
                //! Compare operator.
                //!
                //! \param other The other object.
                //!
                //! Returns true if the other object is equal or bigger and false otherwise.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
                operator<=(const BaseIterator &other) const -> bool
                {
                    return mIndex <= other.mIndex;
                }

                //-----------------------------------------------------------------------------
                //! Compare operator.
                //!
                //! \param other The other object.
                //!
                //! Returns true if the other object is equal or smaller and false
                //! otherwise.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
                operator>=(const BaseIterator &other) const -> bool
                {
                    return mIndex >= other.mIndex;
                }

                //-----------------------------------------------------------------------------
                //! Returns the current element.
                //!
                //! Returns a reference to the current index.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator*() -> T &
                {
                    return *(mData + mIndex);
                }
            };
        }
    }
}