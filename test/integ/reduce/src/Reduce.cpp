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
                typename TIdx = std::uint64_t>
            class TestSetupReduce : public TestSetupBase<TDim, TAcc, TData, TIdx>
            {
            public:
                using TestSetupBase<TDim, TAcc, TData, TIdx>::TestSetupBase;

                using Base = typename vikunja::test::reduce::TestSetupBase<TDim, TAcc, TData, TIdx>;

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
                };
            };

            template<
                typename TDim,
                template<typename, typename>
                class TAcc,
                typename TData,
                typename TIdx = std::uint64_t>
            class TestSetupReduceTransform : public TestSetupBase<TDim, TAcc, TData, TIdx>
            {
            public:
                using TestSetupBase<TDim, TAcc, TData, TIdx>::TestSetupBase;

                using Base = typename vikunja::test::reduce::TestSetupBase<TDim, TAcc, TData, TIdx>;

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
                };
            };

        } // namespace reduce
    } // namespace test
} // namespace vikunja


template<typename TData>
struct Sum
{
    ALPAKA_FN_HOST_ACC TData operator()(TData const i, TData const j) const
    {
        return i + j;
    }
};

template<typename TData>
struct DoubleNum
{
    ALPAKA_FN_HOST_ACC TData operator()(TData const i) const
    {
        return 2 * i;
    }
};

template<typename TAcc, typename TData>
struct Min
{
    ALPAKA_FN_HOST_ACC TData operator()(TAcc const& acc, TData const i, TData const j) const
    {
        return alpaka::math::min(acc, i, j);
    }
};


template<typename TAcc, typename TData>
struct Max
{
    ALPAKA_FN_HOST_ACC TData operator()(TAcc const& acc, TData const i, TData const j) const
    {
        return alpaka::math::max(acc, i, j);
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

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

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

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    Sum<Data> sumOp;

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

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

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

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

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

    Max<alpaka::ExampleDefaultAcc<Dim, std::uint64_t>, Data> max;
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

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

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

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduceTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    auto reduce = [] ALPAKA_FN_HOST_ACC(Data const i, Data const j) -> Data { return i + j; };
    DoubleNum<Data> transform;

    setup.run(reduce, transform);

    Data const n = static_cast<Data>(size);
    Data expectedResult = 2 * (n * (n + 1) / 2);

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduceTransform mixed lambda and operator with acc object",
    "[reduceTransform][mixFunc][acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

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

    Min<alpaka::ExampleDefaultAcc<Dim, std::uint64_t>, Data> reduce;
    auto transform
        = [] ALPAKA_FN_HOST_ACC(alpaka::ExampleDefaultAcc<Dim, std::uint64_t> const& acc, Data const i) -> Data
    {
        // TODO: check why double is not working
        return static_cast<Data>(alpaka::math::sqrt(acc, static_cast<float>(i)));
    };

    setup.run(reduce, transform);

    std::vector<Data> tmp;
    tmp.resize(size);
    std::transform(
        host_mem_ptr,
        host_mem_ptr + size,
        tmp.begin(),
        [](Data const i) -> Data { return static_cast<Data>(std::sqrt(static_cast<float>(i))); });
    Data expectedResult = *std::min_element(tmp.begin(), tmp.end());

    REQUIRE(setup.get_result() == expectedResult);
}
