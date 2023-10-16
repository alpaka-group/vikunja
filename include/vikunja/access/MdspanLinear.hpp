#pragma once

#include <experimental/mdspan>

#include <assert.h>

namespace stdex = std::experimental;

/**
 * @brief Construct submdspan of mdspan. The submdspan has one rank less than the mdspan. The left dimension is fixed
 * to a specific index. The rest of the dimension contains the full range.
 *
 * @tparam TRank Dimension of the new submdspan (needs to be mdspan::rank()-1).
 */
template<int TRank>
struct Construct_Submdspan;

template<>
struct Construct_Submdspan<1>
{
    template<typename TSpan, typename... Types>
    constexpr auto construct(TSpan span, std::size_t const fixed_index_pos, Types... args)
    {
        return stdex::submdspan(span, fixed_index_pos, args...);
    }
};

template<int TRank>
struct Construct_Submdspan
{
    /**
     * @brief Returns the submdspan of a mdspan, with one dimension less.
     *
     * @tparam TSpan Type of the span
     * @tparam Types needs to std::experimental::full_extent_t
     * @param span mdspan from which the submdspan is created
     * @param fixed_index_pos Index postion of the fixed dimension
     * @param args needs to std::experimental::full_extent
     * @return constexpr auto returns a stdex::submdspan
     */
    template<typename TSpan, typename... Types>
    constexpr auto construct(TSpan span, std::size_t const fixed_index_pos, Types... args)
    {
        return Construct_Submdspan<TRank - 1>{}.construct(span, fixed_index_pos, stdex::full_extent, args...);
    }
};

/**
 * @brief Returns a submdspan of mdspan. The submdspan has one rank less than the mdspan. The left dimension is fixed
 * to a specific index. The rest of the dimension contains the full range.
 *
 * @tparam TMDSpan
 * @param span mdspan from which the submdspan is created
 * @param fixed_index_pos Index postion of the fixed dimension
 * @return constexpr auto returns a stdex::submdspan
 */
template<typename TMDSpan>
constexpr auto submdspan_remove_dim(TMDSpan span, std::size_t const fixed_index_pos)
{
    constexpr auto rank = TMDSpan::rank();
    return Construct_Submdspan<rank - 1>{}.construct(span, fixed_index_pos, stdex::full_extent);
}

/**
 * @brief Iterates over all elements of an n dimension mdspan. The iteration order is from the right to the left
 * dimension.
 *
 * @tparam TDim Rank of the mdspan
 */
template<int TDim>
struct Iterate_mdspan;

template<>
struct Iterate_mdspan<1>
{
    template<typename TSpan, typename TFunc>
    void operator()(TSpan input, TSpan output, TFunc& functor)
    {
        assert(input.extent(0) <= output.extent(0));
        for(auto i = 0; i < input.extent(0); ++i)
        {
            output(i) = functor(input(i));
        }
    }
};

template<int TDim>
struct Iterate_mdspan
{
    /**
     * @brief Iterate over all elements of an mdspan and apply the functor on it.
     *
     * @tparam TSpan type of the mdspan's
     * @tparam TFunc type of the functor
     * @param input The input mdspan
     * @param output The output mdspan
     * @param functor The functor
     */
    template<typename TSpan, typename TFunc>
    void operator()(TSpan input, TSpan output, TFunc& functor)
    {
        assert(input.extent(0) <= output.extent(0));

        for(auto i = 0; i < input.extent(0); ++i)
        {
            auto subinput = submdspan_remove_dim(input, i);
            auto suboutput = submdspan_remove_dim(output, i);
            Iterate_mdspan<TSpan::rank() - 1>{}(subinput, suboutput, functor);
        }
    }
};
