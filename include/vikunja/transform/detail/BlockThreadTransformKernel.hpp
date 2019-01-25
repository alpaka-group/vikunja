//
// Created by hauke on 25.01.19.
//

#pragma once

#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>
#include <alpaka/alpaka.hpp>

namespace vikunja {
    namespace transform {
        namespace detail {

            template<uint64_t TBlockSize, typename TMemAccessPolicy, typename TIn, typename TOut>
            struct BlockThreadTransformKernel {

                template<typename TAcc, typename TIdx,
                        typename TInputIterator, typename TOutputIterator, typename TFunc>
                ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                              TInputIterator const &source,
                                              TOutputIterator const &destination,
                                              TIdx const &n,
                                              TFunc const &func) const {
                    vikunja::mem::iterator::PolicyBasedBlockIterator<TMemAccessPolicy, TAcc, TIn> inputIterator(source, acc, n, TBlockSize);
                    vikunja::mem::iterator::PolicyBasedBlockIterator<TMemAccessPolicy, TAcc, TOut> outputIterator(source, acc, n, TBlockSize);

                    while(inputIterator < inputIterator.end()) {
                        *outputIterator = func(*inputIterator);
                        ++inputIterator;
                        ++outputIterator;
                    }
                }
            };
        }
    }
} // vikunja