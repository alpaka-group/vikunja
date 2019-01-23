//
// Created by mewes30 on 23.01.19.
//

#pragma once

#include <alpaka/alpaka.hpp>

namespace vikunja {
    namespace reduce {
        namespace detail {
            template<typename TFunc>
            struct SmallProblemReduceKernel {
                template<typename TAcc, typename TIdx,
                        typename TInputIterator, typename TOutputIterator>
                ALPAKA_FN_ACC void operator()(TAcc const &acc __attribute__((unused)),
                                              TInputIterator const * const source,
                                              TOutputIterator destination,
                                              TIdx const &n,
                                              TFunc func) const {
                    auto tSum = *(source);
                    for(TIdx i(1); i < n; ++i) {
                        tSum = func(tSum, *(source + i));
                    }
                    *destination = tSum;
                }
            };
        } // detail
    } // reduce
} // vikunja