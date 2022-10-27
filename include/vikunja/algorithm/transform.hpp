#pragma once

//#include <alpaka/alpaka.hpp>

#include <vikunja/access/BlockStrategy.hpp>
#include <vikunja/access/MdspanLinear.hpp>
#include <vikunja/operators/operators.hpp>
#include <vikunja/workdiv/BlockBasedWorkDiv.hpp>

#include <experimental/mdspan>

namespace vikunja
{
    namespace device
    {
        // FIXME: I'm only running on a single core CPU :-(
        template<
            typename TAcc,
            typename WorkDivPolicy = vikunja::workdiv::BlockBasedPolicy<TAcc>,
            typename MemAccessPolicy = vikunja::MemAccess::MemAccessPolicy<TAcc>,
            typename TDevAcc,
            typename TQueue,
            typename TData,
            typename TLayoutPolicy,
            typename TAccessorPolicy,
            typename TInputExtend,
            typename TOutputExtend,
            typename TFunc,
            typename TOperator = vikunja::operators::UnaryOp<TAcc, TFunc, TData>>
        void transform(
            TDevAcc& devAcc,
            TQueue& queue,
            std::experimental::mdspan<TData, TInputExtend, TLayoutPolicy, TAccessorPolicy> input,
            std::experimental::mdspan<TData, TOutputExtend, TLayoutPolicy, TAccessorPolicy> output,
            TFunc const& func)
        {
            constexpr auto input_rank = decltype(input)::rank();
            static_assert(input_rank > 0);

            Iterate_mdspan<input_rank>{}(input, output, func);
        }
    } // namespace device
} // namespace vikunja
