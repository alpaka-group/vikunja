/* Copyright 2021 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "reduce_setup.hpp"

#include <vikunja/reduce/reduce.hpp>
#include <vikunja/test/utility.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

namespace vikunja
{
    namespace test
    {
        namespace reduce
        {
            template<
                typename TDim,
                template<typename, typename>
                class TAcc,
                typename TData,
                typename TDataResult = TData,
                typename TIdx = std::uint64_t>
            class TestSetupReduce : public TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>
            {
            public:
                using TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>::TestSetupBase;

                using Base = typename vikunja::test::reduce::TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>;

                template<typename TReduceFunctor>
                void run(TReduceFunctor reduceFunctor)
                {
                    alpaka::memcpy(Base::Base::queueAcc, Base::m_device_mem, Base::m_host_mem, Base::m_extent);

                    Base::m_result = vikunja::reduce::deviceReduce<typename Base::Acc>(
                        Base::devAcc,
                        Base::devHost,
                        Base::Base::queueAcc,
                        Base::m_size,
                        alpaka::getPtrNative(Base::m_device_mem),
                        reduceFunctor);
                }
            };

            template<
                typename TDim,
                template<typename, typename>
                class TAcc,
                typename TData,
                typename TDataResult = TData,
                typename TIdx = std::uint64_t>
            class TestSetupReduceTransform : public TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>
            {
            public:
                using TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>::TestSetupBase;

                using Base = typename vikunja::test::reduce::TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>;

                template<typename TReduceFunctor, typename TTransformFunctor>
                void run(TReduceFunctor reduceFunctor, TTransformFunctor transformFunctor)
                {
                    alpaka::memcpy(Base::Base::queueAcc, Base::m_device_mem, Base::m_host_mem, Base::m_extent);

                    Base::m_result = vikunja::reduce::deviceTransformReduce<typename Base::Acc>(
                        Base::devAcc,
                        Base::devHost,
                        Base::Base::queueAcc,
                        Base::m_size,
                        alpaka::getPtrNative(Base::m_device_mem),
                        transformFunctor,
                        reduceFunctor);
                }
            };

        } // namespace reduce
    } // namespace test
} // namespace vikunja


struct Sum
{
    template<typename TData>
    ALPAKA_FN_HOST_ACC TData operator()(TData const i, TData const j) const
    {
        return i + j;
    }
};


struct DoubleNum
{
    template<typename TData>
    ALPAKA_FN_HOST_ACC TData operator()(TData const i) const
    {
        return 2 * i;
    }
};


struct Min
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_HOST_ACC TData operator()(TAcc const& acc, TData const i, TData const j) const
    {
        return alpaka::math::min(acc, i, j);
    }
};


struct Max
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_HOST_ACC TData operator()(TAcc const& acc, TData const i, TData const j) const
    {
        return alpaka::math::max(acc, i, j);
    }
};


struct MakePairUnaryOp
{
    template<typename TData>
    ALPAKA_FN_HOST_ACC std::pair<TData, TData> operator()(TData const& x) const
    {
        return std::pair<TData, TData>(x, x);
    }
};


struct MinMaxPairBinaryOp
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_HOST_ACC std::pair<TData, TData> operator()(
        TAcc const& acc,
        std::pair<TData, TData> const& x,
        std::pair<TData, TData> const& y) const
    {
        return std::pair<TData, TData>(
            alpaka::math::min(acc, x.first, y.first),
            alpaka::math::max(acc, x.second, y.second));
    }
};


struct MinMaxPairBinaryOpStd
{
    template<typename TData>
    std::pair<TData, TData> operator()(std::pair<TData, TData> const& x, std::pair<TData, TData> const& y) const
    {
        return std::pair<TData, TData>(std::min(x.first, y.first), std::max(x.second, y.second));
    }
};

struct Sqrt
{
    template<typename TAcc, typename TData>
    TData ALPAKA_FN_HOST_ACC operator()(TAcc const& acc, TData const i) const
    {
        return static_cast<TData>(alpaka::math::sqrt(acc, static_cast<double>(i)));
    }
};

TEMPLATE_TEST_CASE(
    "Test reduce lambda",
    "[reduce][lambda][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    auto reduce = [] ALPAKA_FN_HOST_ACC(Data const i, Data const j) { return i + j; };
    setup.run(reduce);

    Data const n = static_cast<Data>(size);
    Data expectedResult = (n * (n + 1) / 2);

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduce operator",
    "[reduce][operator][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    Sum sumOp;

    setup.run(sumOp);

    Data const n = static_cast<Data>(size);
    Data expectedResult = (n * (n + 1) / 2);

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduce lambda with acc object",
    "[reduce][lambda][acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr,
        host_mem_ptr + size,
        [&distribution, &generator]() { return distribution(generator); });

    auto reduce
        = [] ALPAKA_FN_HOST_ACC(alpaka::ExampleDefaultAcc<Dim, std::uint64_t> const& acc, Data const i, Data const j)
    { return alpaka::math::max(acc, i, j); };
    setup.run(reduce);

    Data expectedResult = *std::max_element(host_mem_ptr, host_mem_ptr + size);

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduce operator with acc object",
    "[reduce][operator][acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr,
        host_mem_ptr + size,
        [&distribution, &generator]() { return distribution(generator); });

    Max max;
    setup.run(max);

    Data expectedResult = *std::max_element(host_mem_ptr, host_mem_ptr + size);

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduceTransform lambda",
    "[reduceTransform][lambda][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::reduce::TestSetupReduceTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    auto reduce = [] ALPAKA_FN_HOST_ACC(Data const i, Data const j) { return i + j; };
    auto transform = [] ALPAKA_FN_HOST_ACC(Data const i) { return 2 * i; };

    setup.run(reduce, transform);

    Data const n = static_cast<Data>(size);
    Data expectedResult = 2 * (n * (n + 1) / 2);

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduceTransform mix function and operator",
    "[reduceTransform][mixFunc][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::reduce::TestSetupReduceTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    auto reduce = [] ALPAKA_FN_HOST_ACC(Data const i, Data const j) -> Data { return i + j; };
    DoubleNum transform;

    setup.run(reduce, transform);

    Data const n = static_cast<Data>(size);
    Data expectedResult = 2 * (n * (n + 1) / 2);

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduceTransform mixed lambda and operator with acc object",
    "[reduceTransform][operator][acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::reduce::TestSetupReduceTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr,
        host_mem_ptr + size,
        [&distribution, &generator]() { return distribution(generator); });

    Min reduce;
    Sqrt transform;

    setup.run(reduce, transform);

    std::vector<Data> tmp;
    tmp.resize(size);
    std::transform(
        host_mem_ptr,
        host_mem_ptr + size,
        tmp.begin(),
        [](Data const i) -> Data { return static_cast<Data>(std::sqrt(static_cast<double>(i))); });
    Data expectedResult = *std::min_element(tmp.begin(), tmp.end());

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduce with operators which uses std::pair",
    "[reduce][operator][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using PairType = float;
    using Data = std::pair<PairType, PairType>;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::uniform_real_distribution<PairType> distribution(
        std::numeric_limits<PairType>::min(),
        std::numeric_limits<PairType>::max());
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr,
        host_mem_ptr + size,
        [&distribution, &generator]()
        { return std::make_pair<PairType, PairType>(distribution(generator), distribution(generator)); });


    MinMaxPairBinaryOp reduce;

    setup.run(reduce);

    MinMaxPairBinaryOpStd reduceStd;

    Data expectedResult = std::accumulate(host_mem_ptr, host_mem_ptr + size, host_mem_ptr[0], reduceStd);

    REQUIRE(setup.get_result().first == expectedResult.first);
    REQUIRE(setup.get_result().second == expectedResult.second);
}

TEMPLATE_TEST_CASE(
    "Test reduceTransform with operators which uses as input Data and expect as result std::pair<Data, Data>",
    "[reduceTransform][operator][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = float;
    using ReturnType = std::pair<Data, Data>;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::reduce::TestSetupReduceTransform<Dim, alpaka::ExampleDefaultAcc, Data, ReturnType> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::uniform_real_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr,
        host_mem_ptr + size,
        [&distribution, &generator]() { return distribution(generator); });

    MinMaxPairBinaryOp reduce;
    MakePairUnaryOp transform;

    setup.run(reduce, transform);

    // calculate result
    std::vector<ReturnType> tmp;
    tmp.resize(size);
    std::transform(host_mem_ptr, host_mem_ptr + size, tmp.begin(), transform);

    MinMaxPairBinaryOpStd reduceStd;
    ReturnType expectedResult = std::accumulate(tmp.begin(), tmp.end(), tmp[0], reduceStd);

    // compare result
    REQUIRE(setup.get_result().first == expectedResult.first);
    REQUIRE(setup.get_result().second == expectedResult.second);
}
