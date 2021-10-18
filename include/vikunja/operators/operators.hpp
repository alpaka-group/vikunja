/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

// original source: https://newbedev.com/count-the-number-of-arguments-in-a-lambda

#pragma once

#include <type_traits>
#include <utility>
#include <functional>
#include <alpaka/alpaka.hpp>


namespace vikunjaStd
{
#ifdef __cpp_lib_is_invocable
    using std::is_invocable;
#else
    // define std::is_invocable for c++14 an below
    template<typename F, typename... Args>
    struct is_invocable
        : std::is_constructible<
              std::function<void(Args...)>,
              std::reference_wrapper<typename std::remove_reference<F>::type>>
    {
    };

#endif
} // namespace vikunjaStd

namespace vikunja
{
    /**
     * The vikunja::operators namespace contains type traits for the functors that are passed to the vikunja functions.
     The type traits have two tasks.
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
        struct UnaryOp
        {
        };


        /**
         * Operator trait for unary functors.
         * Accepts: func(TData input) and func(TAcc acc, TData input)
         * tparam TAcc Alpaka TAcc type.
         * tparam TFunc Type of the functor.
         * tparam TData Type of the functor data argument.
         */
        template<typename TAcc, typename TFunc, typename TData>
        struct UnaryOp<TAcc, TFunc, TData, std::enable_if_t<vikunjaStd::is_invocable<TFunc&, TData>::value>>
        {
            /**
             * return type of the functor.
             */
            using TRed = typename std::result_of<TFunc(TData)>::type;

            /**
             * Execute the functor with a data argument. acc object is not injected.
             * param f functor
             * param arg argument, which is passed to the functor
             */
            ALPAKA_FN_HOST_ACC inline static TRed run(TAcc const&, TFunc f, TData arg)
            {
                return f(arg);
            }
        };

        /**
         * Operator trait for unary functors.
         * Accepts: func(TData input) and func(TAcc acc, TData input)
         * tparam TAcc Alpaka TAcc type.
         * tparam TFunc Type of the functor.
         * tparam TData Type of the functor data argument.
         */
        template<typename TAcc, typename TFunc, typename TData>
        struct UnaryOp<TAcc, TFunc, TData, std::enable_if_t<vikunjaStd::is_invocable<TFunc&, TAcc, TData>::value>>
        {
            /**
             * return type of the functor.
             */
            using TRed = typename std::result_of<TFunc(TAcc, TData)>::type;

            /**
             * Execute the functor with a data argument. acc object is injected.
             * param acc alpaka acc object
             * param f functor
             * param arg argument, which is passed to the functor
             */
            ALPAKA_FN_HOST_ACC inline static TRed run(TAcc const& acc, TFunc f, TData arg)
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
        struct BinaryOp
        {
        };


        /**
         * Operator trait for binary functors.
         * Accepts: func(TData1 input1, TData2 input2) and func(TAcc acc, TData1 input1, TData2 input2)
         * tparam TAcc Alpaka TAcc type.
         * tparam TFunc Type of the functor.
         * tparam TData1 Type of the first functor data argument.
         * tparam TData2 Type of the second functor data argument.
         */
        template<typename TAcc, typename TFunc, typename TData1, typename TData2>
        struct BinaryOp<
            TAcc,
            TFunc,
            TData1,
            TData2,
            std::enable_if_t<vikunjaStd::is_invocable<TFunc&, TData1, TData2>::value>>
        {
            /**
             * return type of the functor.
             */
            using TRed = typename std::result_of<TFunc(TData1, TData2)>::type;


            /**
             * Execute the functor with two data argument. acc object is not injected.
             * param f functor
             * param arg1 first argument, which is passed to the functor
             * param arg2 second argument, which is passed to the functor
             */
            ALPAKA_FN_HOST_ACC inline static TRed run(TAcc const&, TFunc f, TData1 arg1, TData2 arg2)
            {
                return f(arg1, arg2);
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
        template<typename TAcc, typename TFunc, typename TData1, typename TData2>
        struct BinaryOp<
            TAcc,
            TFunc,
            TData1,
            TData2,
            std::enable_if_t<vikunjaStd::is_invocable<TFunc&, TAcc, TData1, TData2>::value>>
        {
            /**
             * return type of the functor.
             */
            using TRed = typename std::result_of<TFunc(TAcc, TData1, TData2)>::type;

            /**
             * Execute the functor with two data argument. acc object is injected.
             * param acc alpaka acc object
             * param f functor
             * param arg1 first argument, which is passed to the functor
             * param arg2 second argument, which is passed to the functor
             */
            ALPAKA_FN_HOST_ACC inline static TRed run(TAcc const& acc, TFunc f, TData1 arg1, TData2 arg2)
            {
                return f(acc, arg1, arg2);
            }
        };

    } // namespace operators
} // namespace vikunja
