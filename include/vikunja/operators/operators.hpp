/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <functional>
#include <type_traits>
#include <utility>


namespace vikunja
{
    /**
     * The vikunja::operators namespace contains type traits for the functors that are passed to the vikunja functions.
     The type traits have two tasks:
     * 1. Limit the number of data arguments of the functors for a given function, e.g. vikunja::transform allows only
     functors with one data argument.
     * 2. Provide a functor interface with an optional alpaka TAcc object. For example, a unary operator has an
     interface with one argument (single data argument) or two arguments (TAcc argument and data argument)
     */
    namespace operators
    {
        /**
         * Operator trait for unary functors.
         * Accepts: func(TData input) and func(TAcc acc, TData input)
         * tparam TAcc Alpaka TAcc type.
         * tparam TFunc Type of the functor.
         * tparam TData Type of the functor data argument.
         */
        template<typename TAcc, typename TFunc, typename TData, typename TSfinae = void>
        struct UnaryOp;

        template<typename TFunc, typename TData>
        using enable_if_UnaryOp_without_TAcc
            = std::enable_if_t<std::is_invocable<decltype(std::declval<TFunc>()), TData>::value>;

        /**
         * Operator trait for unary functors.
         * Accepts: func(TData input) and func(TAcc acc, TData input)
         * tparam TAcc Alpaka TAcc type.
         * tparam TFunc Type of the functor.
         * tparam TData Type of the functor data argument.
         */
        template<typename TAcc, typename TFunc, typename TData>
        struct UnaryOp<TAcc, TFunc, TData, enable_if_UnaryOp_without_TAcc<TFunc, TData>>
        {
            /**
             * return type of the functor.
             */
            using TRed = typename std::result_of<decltype(std::declval<TFunc>())(TData)>::type;

            /**
             * Execute the functor with a data argument. acc object is not injected.
             * param f functor
             * param arg argument, which is passed to the functor
             */
            ALPAKA_FN_HOST_ACC inline static TRed run(TAcc const&, TFunc f, TData const arg)
            {
                return f(arg);
            }
        };

        template<typename TFunc, typename TAcc, typename TData>
        using enable_if_UnaryOp_with_TAcc
            = std::enable_if_t<std::is_invocable<decltype(std::declval<TFunc>()), TAcc, TData>::value>;

        /**
         * Operator trait for unary functors.
         * Accepts: func(TData input) and func(TAcc acc, TData input)
         * tparam TAcc Alpaka TAcc type.
         * tparam TFunc Type of the functor.
         * tparam TData Type of the functor data argument.
         */
        template<typename TAcc, typename TFunc, typename TData>
        struct UnaryOp<TAcc, TFunc, TData, enable_if_UnaryOp_with_TAcc<TFunc, TAcc, TData>>
        {
            /**
             * return type of the functor.
             */
            using TRed = typename std::result_of<decltype(std::declval<TFunc>())(TAcc, TData)>::type;

            /**
             * Execute the functor with a data argument. acc object is injected.
             * param acc alpaka acc object
             * param f functor
             * param arg argument, which is passed to the functor
             */
            ALPAKA_FN_HOST_ACC inline static TRed run(TAcc const& acc, TFunc f, TData const arg)
            {
                return f(acc, arg);
            }
        };


        /**
         * Operator trait for binary functors.
         * Accepts: func(TData1 input1, TData2 input2) and func(TAcc acc, TData1 input1, TData2 input2)
         * tparam TAcc Alpaka TAcc type.
         * tparam TFunc Type of the functor.
         * tparam TData1 Type of the first functor data argument.
         * tparam TData2 Type of the second functor data argument.
         */
        template<typename TAcc, typename TFunc, typename TData1, typename TData2, typename TSfinae = void>
        struct BinaryOp;

        template<typename TFunc, typename TData1, typename TData2>
        using enable_if_BinaryOp_without_TAcc
            = std::enable_if_t<std::is_invocable<decltype(std::declval<TFunc>()), TData1, TData2>::value>;

        /**
         * Operator trait for binary functors.
         * Accepts: func(TData1 input1, TData2 input2) and func(TAcc acc, TData1 input1, TData2 input2)
         * tparam TAcc Alpaka TAcc type.
         * tparam TFunc Type of the functor.
         * tparam TData1 Type of the first functor data argument.
         * tparam TData2 Type of the second functor data argument.
         */
        template<typename TAcc, typename TFunc, typename TData1, typename TData2>
        struct BinaryOp<TAcc, TFunc, TData1, TData2, enable_if_BinaryOp_without_TAcc<TFunc, TData1, TData2>>
        {
            /**
             * return type of the functor.
             */
            using TRed = typename std::result_of<decltype(std::declval<TFunc>())(TData1, TData2)>::type;


            /**
             * Execute the functor with two data argument. acc object is not injected.
             * param f functor
             * param arg1 first argument, which is passed to the functor
             * param arg2 second argument, which is passed to the functor
             */
            ALPAKA_FN_HOST_ACC inline static TRed run(TAcc const&, TFunc f, TData1 const arg1, TData2 const arg2)
            {
                return f(arg1, arg2);
            }
        };

        template<typename TFunc, typename TAcc, typename TData1, typename TData2>
        using enable_if_BinaryOp_with_TAcc
            = std::enable_if_t<std::is_invocable<decltype(std::declval<TFunc>()), TAcc, TData1, TData2>::value>;

        /**
         * Operator trait for binary functors.
         * Accepts: func(TData1 input1, TData2 input2) and func(TAcc acc, TData1 input1, TData2 input2)
         * tparam TAcc Alpaka TAcc type.
         * tparam TFunc Type of the functor.
         * tparam TData1 Type of the first functor data argument.
         * tparam TData2 Type of the second functor data argument.
         */
        template<typename TAcc, typename TFunc, typename TData1, typename TData2>
        struct BinaryOp<TAcc, TFunc, TData1, TData2, enable_if_BinaryOp_with_TAcc<TFunc, TAcc, TData1, TData2>>
        {
            /**
             * return type of the functor.
             */
            using TRed = typename std::result_of<decltype(std::declval<TFunc>())(TAcc, TData1, TData2)>::type;

            /**
             * Execute the functor with two data argument. acc object is injected.
             * param acc alpaka acc object
             * param f functor
             * param arg1 first argument, which is passed to the functor
             * param arg2 second argument, which is passed to the functor
             */
            ALPAKA_FN_HOST_ACC inline static TRed run(TAcc const& acc, TFunc f, TData1 const arg1, TData2 const arg2)
            {
                return f(acc, arg1, arg2);
            }
        };

    } // namespace operators
} // namespace vikunja
