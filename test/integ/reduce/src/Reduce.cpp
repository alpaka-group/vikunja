/* Copyright 2021 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <vikunja/reduce/reduce.hpp>
#include <catch2/catch.hpp>
#include <numeric>
#include <limits>
#include <random>
#include <algorithm>
#include <sstream>

#include "reduce_setup.hpp"

template<typename TDim, typename TData>
inline std::string print_info(TData const size)
{
    std::stringstream strs;

    using Acc = alpaka::ExampleDefaultAcc<TDim, std::uint64_t>;
    strs << "Testing accelerator: " << alpaka::getAccName<Acc>() << " with size: " << size << "\n";

    using MemAccess = vikunja::mem::iterator::MemAccessPolicy<Acc>;
    strs << "MemAccessPolicy: " << MemAccess::getName() << "\n";

    return strs.str();
}

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
                TestSetupReduce(uint64_t const memSize) : TestSetupBase<TDim, TAcc, TData, TIdx>(memSize)
                {
                }

                using P = typename vikunja::test::reduce::TestSetupBase<TDim, TAcc, TData, TIdx>;

                template<typename TReduceFunctor>
                void run(TReduceFunctor reduceFunctor)
                {
                    alpaka::memcpy(P::P::queueAcc, P::m_device_mem, P::m_host_mem, P::m_extent);

                    P::m_result = vikunja::reduce::deviceReduce<typename P::Acc>(
                        P::devAcc,
                        P::devHost,
                        P::P::queueAcc,
                        P::m_size,
                        alpaka::getPtrNative(P::m_device_mem),
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
                TestSetupReduceTransform(uint64_t const memSize) : TestSetupBase<TDim, TAcc, TData, TIdx>(memSize)
                {
                }

                using P = typename vikunja::test::reduce::TestSetupBase<TDim, TAcc, TData, TIdx>;

                template<typename TReduceFunctor, typename TTransformFunctor>
                void run(TReduceFunctor reduceFunctor, TTransformFunctor transformFunctor)
                {
                    alpaka::memcpy(P::P::queueAcc, P::m_device_mem, P::m_host_mem, P::m_extent);

                    P::m_result = vikunja::reduce::deviceTransformReduce<typename P::Acc>(
                        P::devAcc,
                        P::devHost,
                        P::P::queueAcc,
                        P::m_size,
                        alpaka::getPtrNative(P::m_device_mem),
                        transformFunctor,
                        reduceFunctor);
                };
            };

        } // namespace reduce
    } // namespace test
} // namespace vikunja


template<typename TData>
ALPAKA_FN_HOST_ACC TData sum(TData const i, TData const j)
{
    return i + j;
}

template<typename TData>
ALPAKA_FN_HOST_ACC TData doubleNum(TData const i)
{
    return 2 * i;
}

template<typename TAcc, typename TData>
ALPAKA_FN_HOST_ACC TData min(TAcc const& acc, TData const i, TData const j)
{
    return alpaka::math::min(acc, i, j);
};

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
struct Max
{
    ALPAKA_FN_HOST_ACC TData operator()(TAcc const& acc, TData const i, TData const j) const
    {
        return alpaka::math::max(acc, i, j);
    }
};


TEMPLATE_TEST_CASE(
    "Test reduce function",
    "[reduce][function][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((print_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    setup.run(sum<Data>);

    Data const n = static_cast<Data>(size);
    Data expectedResult = (n * (n + 1) / 2);

    REQUIRE(setup.get_result() == expectedResult);
}

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

    INFO((print_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    auto reduce = [] ALPAKA_FN_HOST_ACC(Data i, Data j) { return i + j; };
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

    INFO((print_info<Dim, Data>(size)));

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

    INFO((print_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(host_mem_ptr, host_mem_ptr + size, [&distribution, &generator]() {
        return distribution(generator);
    });

    auto reduce = [] ALPAKA_FN_HOST_ACC(alpaka::ExampleDefaultAcc<Dim, std::uint64_t> const& acc, Data i, Data j) {
        return alpaka::math::max(acc, i, j);
    };
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

    INFO((print_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduce<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(host_mem_ptr, host_mem_ptr + size, [&distribution, &generator]() {
        return distribution(generator);
    });

    Max<alpaka::ExampleDefaultAcc<Dim, std::uint64_t>, Data> max;
    setup.run(max);

    Data expectedResult = *std::max_element(host_mem_ptr, host_mem_ptr + size);

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduceTransform function",
    "[reduceTransform][function][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((print_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduceTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    setup.run(sum<Data>, doubleNum<Data>);

    Data const n = static_cast<Data>(size);
    Data expectedResult = 2 * (n * (n + 1) / 2);

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

    INFO((print_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduceTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    auto reduce = [] ALPAKA_FN_HOST_ACC(Data i, Data j) { return i + j; };
    auto transform = [=] ALPAKA_FN_HOST_ACC(Data i) { return 2 * i; };

    setup.run(reduce, transform);

    Data const n = static_cast<Data>(size);
    Data expectedResult = 2 * (n * (n + 1) / 2);

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduceTransform mixed lambda and function with acc object",
    "[reduceTransform][mixFunc][acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((print_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduceTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(host_mem_ptr, host_mem_ptr + size, [&distribution, &generator]() {
        return distribution(generator);
    });

    auto transform
        = [] ALPAKA_FN_HOST_ACC(alpaka::ExampleDefaultAcc<Dim, std::uint64_t> const& acc, Data const i) -> Data {
        // TODO: check why double is not working
        return static_cast<Data>(alpaka::math::sqrt(acc, static_cast<float>(i)));
    };

    setup.run(min<alpaka::ExampleDefaultAcc<Dim, std::uint64_t>, Data>, transform);

    std::vector<Data> tmp;
    tmp.resize(size);
    std::transform(host_mem_ptr, host_mem_ptr + size, tmp.begin(), [](Data const i) -> Data {
        return static_cast<Data>(std::sqrt(static_cast<float>(i)));
    });
    Data expectedResult = *std::min_element(tmp.begin(), tmp.end());

    REQUIRE(setup.get_result() == expectedResult);
}

TEMPLATE_TEST_CASE(
    "Test reduceTransform mix function operator",
    "[reduceTransform][mixFunc][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((print_info<Dim, Data>(size)));

    vikunja::test::reduce::TestSetupReduceTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    DoubleNum<Data> transform;

    setup.run(sum<Data>, transform);

    Data const n = static_cast<Data>(size);
    Data expectedResult = 2 * (n * (n + 1) / 2);

    REQUIRE(setup.get_result() == expectedResult);
}
