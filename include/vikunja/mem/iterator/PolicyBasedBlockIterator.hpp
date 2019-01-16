//
// Created by mewes30 on 16.01.19.
//

#pragma once

#include "BaseIterator.hpp"
#include <alpaka/alpaka.hpp>

namespace vikunja {
    namespace mem {
        namespace iterator {
            template <typename MemAccessPolicy, typename TAcc, typename T, typename TBuf = T>
            class PolicyBasedBlockIterator : public BaseIterator<T, TBuf> {
            private:
                uint64_t mStep;
            public:
                
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PolicyBasedBlockIterator(const TBuf * data, TAcc const &acc, uint64_t problemSize, uint64_t blockSize) : BaseIterator<T, TBuf>(data, MemAccessPolicy::getStartIndex(acc, problemSize, blockSize), MemAccessPolicy::getEndIndex(acc, problemSize, blockSize)), mStep(MemAccessPolicy::getStepSize(acc, problemSize, blockSize))
                {}

                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PolicyBasedBlockIterator(const PolicyBasedBlockIterator &other) = default;
                
                //-----------------------------------------------------------------------------
                //! Returns the iterator for the last item.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto end() const -> PolicyBasedBlockIterator
                {
                    PolicyBasedBlockIterator ret(*this);
                    ret.mIndex = this->mMaximum;
                    return ret;
                }

                //-----------------------------------------------------------------------------
                //! Increments the internal pointer to the next one and returns this
                //! element.
                //!
                //! Returns a reference to the next index.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++() -> PolicyBasedBlockIterator &
                {
                    this->mIndex += this->mStep;
                    return *this;
                }

                //-----------------------------------------------------------------------------
                //! Returns the current element and increments the internal pointer to the
                //! next one.
                //!
                //! Returns a reference to the current index.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++(int) -> PolicyBasedBlockIterator
                {
                    auto ret(*this);
                    this->mIndex += this->mStep;
                    return ret;
                }

                //-----------------------------------------------------------------------------
                //! Decrements the internal pointer to the previous one and returns the this
                //! element.
                //!
                //! Returns a reference to the previous index.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--() -> PolicyBasedBlockIterator &
                {
                    this->mIndex -= this->mStep;
                    return *this;
                }

                //-----------------------------------------------------------------------------
                //! Returns the current element and decrements the internal pointer to the
                //! previous one.
                //!
                //! Returns a reference to the current index.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--(int) -> PolicyBasedBlockIterator
                {
                    auto ret(*this);
                    this->mIndex -= this->mStep;
                    return ret;
                }

                //-----------------------------------------------------------------------------
                //! Returns the index + a supplied offset.
                //!
                //! \param n The offset.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+(uint64_t n) const
                -> PolicyBasedBlockIterator
                {
                    auto ret(*this);
                    ret.mIndex += n * mStep;
                    return ret;
                }

                //-----------------------------------------------------------------------------
                //! Returns the index - a supplied offset.
                //!
                //! \param n The offset.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-(uint64_t n) const
                -> PolicyBasedBlockIterator
                {
                    auto ret(*this);
                    ret.mIndex -= n * mStep;
                    return ret;
                }

                //-----------------------------------------------------------------------------
                //! Addition assignment.
                //!
                //! \param offset The offset.
                //!
                //! Returns the current object offset by the offset.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+=(uint64_t offset)
                -> PolicyBasedBlockIterator &
                {
                    this->mIndex += offset * this->mStep;
                    return *this;
                }

                //-----------------------------------------------------------------------------
                //! Substraction assignment.
                //!
                //! \param offset The offset.
                //!
                //! Returns the current object offset by the offset.
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-=(uint64_t offset)
                -> PolicyBasedBlockIterator &
                {
                    this->mIndex -= offset * this->mStep;
                    return *this;
                }
            };
            
            namespace policies {
                struct LinearMemAccessPolicy {
                    template<typename TAcc, typename TIdx>
                    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
                    static auto getStartIndex(TAcc const &acc, TIdx const &problemSize, TIdx const &blockSize) -> TIdx const {
                        auto gridDimension = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
                        auto indexInBlock = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                        auto gridSize = gridDimension * blockSize;
                        // TODO: catch overflow
                        return (problemSize * indexInBlock) / gridSize;
                    }

                    template<typename TAcc, typename TIdx>
                    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
                    static auto getEndIndex(TAcc const &acc, TIdx const &problemSize, TIdx const &blockSize) -> TIdx const {
                        auto gridDimension = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
                        auto indexInBlock = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                        auto gridSize = gridDimension * blockSize;
                        // TODO: catch overflow
                        return (problemSize * indexInBlock + problemSize) / gridSize;
                    }

                    template<typename TAcc, typename TIdx>
                    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
                    static constexpr auto getStepSize(TAcc const &acc, TIdx const &problemSize, TIdx const &blockSize) -> TIdx const {
                        return 1;
                    }

                    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
                    static constexpr auto getName() -> char * {
                        return const_cast<char *>("LinearMemAccessPolicy");
                    }
                };

                struct GridStridingMemAccessPolicy {
                    template<typename TAcc, typename TIdx>
                    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
                    static auto getStartIndex(TAcc const &acc, TIdx const &problemSize, TIdx const &blockSize) -> TIdx const {
                        return alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                    }

                    template<typename TAcc, typename TIdx>
                    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
                    static auto getEndIndex(TAcc const &acc, TIdx const &problemSize, TIdx const &blockSize) -> TIdx const {
                        return problemSize;
                    }

                    template<typename TAcc, typename TIdx>
                    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
                    static auto getStepSize(TAcc const &acc, TIdx const &problemSize, TIdx const &blockSize) -> TIdx const {
                        auto gridDimension = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
                        return gridDimension * blockSize;
                    }

                    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
                    static constexpr auto getName() -> char * {
                        return const_cast<char *>("GridStridingMemAccessPolicy");
                    }
                };
            } // policies

            namespace traits {
                template<typename TAcc, typename TSfinae = void>
                struct GetMemAccessPolicyByPltf{};

                template<>
                struct GetMemAccessPolicyByPltf<alpaka::pltf::PltfCpu> {
                    using type = policies::LinearMemAccessPolicy;
                };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                template<>
                struct GetMemAccessPolicyByPltf<alpaka::pltf::PltfCudaRt> {
                    using type = policies::GridStridingMemAccessPolicy;
                };
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

            } //traits

            // shortcut to derive policy from accelerator
            template<typename TAcc>
            using MemAccessPolicy = typename traits::GetMemAccessPolicyByPltf<alpaka::pltf::Pltf<alpaka::dev::Dev<TAcc>>>::type;
        } // iterator
    } // mem
} // vikunja