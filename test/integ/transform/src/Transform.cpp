/* Copyright 2021 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "transform_setup.hpp"

#include <vikunja/test/utility.hpp>
#include <vikunja/transform/transform.hpp>

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
        namespace transform
        {
            template<
                typename TDim,
                template<typename, typename>
                class TAcc,
                typename TData,
                typename TIdx = std::uint64_t>
            class TestSetupTransform : public TestSetupBase<TDim, TAcc, TData, TIdx>
            {
            public:
                using TestSetupBase<TDim, TAcc, TData, TIdx>::TestSetupBase;

                using Base = typename vikunja::test::transform::TestSetupBase<TDim, TAcc, TData, TIdx>;

                template<typename TReduceFunctor>
                void run(TReduceFunctor reduceFunctor)
                {
                    alpaka::memcpy(
                        Base::Base::queueAcc,
                        Base::m_device_input1_mem,
                        Base::m_host_input1_mem,
                        Base::m_extent);

                    vikunja::transform::deviceTransform<typename Base::Acc>(
                        Base::devAcc,
                        Base::Base::queueAcc,
                        Base::m_size,
                        alpaka::getPtrNative(Base::m_device_input1_mem),
                        alpaka::getPtrNative(Base::m_device_output_mem),
                        reduceFunctor);

                    alpaka::memcpy(
                        Base::Base::queueAcc,
                        Base::m_host_output_mem,
                        Base::m_device_output_mem,
                        Base::m_extent);
                };
            };

            template<
                typename TDim,
                template<typename, typename>
                class TAcc,
                typename TData,
                typename TIdx = std::uint64_t>
            class TestSetupTransformDoubleInput : public TestSetupBase<TDim, TAcc, TData, TIdx>
            {
            private:
                using Base = typename vikunja::test::transform::TestSetupBase<TDim, TAcc, TData, TIdx>;
                using BufHost = typename Base::BufHost;
                using BufDev = typename Base::BufDev;

                BufHost m_host_input2_mem;
                BufDev m_device_input2_mem;

            public:
                TestSetupTransformDoubleInput(uint64_t const memSize)
                    : TestSetupBase<TDim, TAcc, TData, TIdx>(memSize)
                    , m_host_input2_mem(alpaka::allocBuf<TData, TIdx>(Base::Base::devHost, Base::m_extent))
                    , m_device_input2_mem(alpaka::allocBuf<TData, TIdx>(Base::Base::devAcc, Base::m_extent))
                {
                }


                TData* get_host_input2_mem_ptr()
                {
                    return alpaka::getPtrNative(m_host_input2_mem);
                }

                template<typename TReduceFunctor>
                void run(TReduceFunctor reduceFunctor)
                {
                    alpaka::memcpy(
                        Base::Base::queueAcc,
                        Base::m_device_input1_mem,
                        Base::m_host_input1_mem,
                        Base::m_extent);
                    alpaka::memcpy(Base::Base::queueAcc, m_device_input2_mem, m_host_input2_mem, Base::m_extent);

                    vikunja::transform::deviceTransform<typename Base::Acc>(
                        Base::devAcc,
                        Base::Base::queueAcc,
                        Base::m_size,
                        alpaka::getPtrNative(Base::m_device_input1_mem),
                        alpaka::getPtrNative(m_device_input2_mem),
                        alpaka::getPtrNative(Base::m_device_output_mem),
                        reduceFunctor);

                    alpaka::memcpy(
                        Base::Base::queueAcc,
                        Base::m_host_output_mem,
                        Base::m_device_output_mem,
                        Base::m_extent);
                };
            };

        } // namespace transform
    } // namespace test
} // namespace vikunja


template<typename TData>
struct IncOne
{
    ALPAKA_FN_HOST_ACC TData operator()(TData const val) const
    {
        return val + 1;
    }
};

template<typename TAcc, typename TData, int TMaxValue>
struct Max
{
    ALPAKA_FN_HOST_ACC TData operator()(TAcc const& acc, TData const i) const
    {
        return alpaka::math::max(acc, i, TMaxValue);
    }
};

template<typename TData>
struct MathOperator
{
    ALPAKA_FN_HOST_ACC TData operator()(TData const i, TData const j) const
    {
        return ((i * 2) - (i + 1)) + ((j * 2) - (j + 1));
    }
};


TEMPLATE_TEST_CASE(
    "Test transform with lambda but without acc object",
    "[transform][lambda][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

    vikunja::test::transform::TestSetupTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_input1_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    auto transform = [] ALPAKA_FN_HOST_ACC(Data const i) { return i + 2; };

    setup.run(transform);

    std::vector<Data> expected_result;
    expected_result.resize(size);
    std::iota(expected_result.begin(), expected_result.end(), 3);

    Data const* const result_ptr = setup.get_host_output_mem_ptr();
    std::vector<Data> result(result_ptr, result_ptr + size);

    REQUIRE(result == expected_result);
}

TEMPLATE_TEST_CASE(
    "Test transform with operator but without acc object",
    "[transform][operator][noAcc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

    vikunja::test::transform::TestSetupTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_input1_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    IncOne<Data> transform;

    setup.run(transform);

    std::vector<Data> expected_result;
    expected_result.resize(size);
    std::iota(expected_result.begin(), expected_result.end(), 2);

    Data const* const result_ptr = setup.get_host_output_mem_ptr();
    std::vector<Data> result(result_ptr, result_ptr + size);

    REQUIRE(result == expected_result);
}

TEMPLATE_TEST_CASE(
    "Test transform with lambda and acc object",
    "[transform][lambda][Acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

    vikunja::test::transform::TestSetupTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_input1_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr,
        host_mem_ptr + size,
        [&distribution, &generator]() { return distribution(generator); });

    auto transform = [] ALPAKA_FN_HOST_ACC(alpaka::ExampleDefaultAcc<Dim, std::uint64_t> const& acc, Data const i)
    { return alpaka::math::min(acc, i, static_cast<Data>(1 << 10)); };

    setup.run(transform);

    std::vector<Data> expected_result;
    expected_result.resize(size);
    std::transform(
        host_mem_ptr,
        host_mem_ptr + size,
        expected_result.begin(),
        [](Data const i) { return std::min(i, static_cast<Data>(1 << 10)); });

    Data const* const result_ptr = setup.get_host_output_mem_ptr();
    std::vector<Data> result(result_ptr, result_ptr + size);

    REQUIRE(result == expected_result);
}

TEMPLATE_TEST_CASE(
    "Test transform with operator and acc object",
    "[transform][operator][Acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = int;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

    vikunja::test::transform::TestSetupTransform<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_input1_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr,
        host_mem_ptr + size,
        [&distribution, &generator]() { return distribution(generator); });

    Max<alpaka::ExampleDefaultAcc<Dim, std::uint64_t>, Data, (1 << 10)> transform;

    setup.run(transform);

    std::vector<Data> expected_result;
    expected_result.resize(size);
    std::transform(
        host_mem_ptr,
        host_mem_ptr + size,
        expected_result.begin(),
        [](Data const i) { return std::max(i, static_cast<Data>(1 << 10)); });

    Data const* const result_ptr = setup.get_host_output_mem_ptr();
    std::vector<Data> result(result_ptr, result_ptr + size);

    REQUIRE(result == expected_result);
}

TEMPLATE_TEST_CASE(
    "Test transform double input with operator but without acc object",
    "[transform][operator][Acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

    vikunja::test::transform::TestSetupTransformDoubleInput<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr1 = setup.get_host_input1_mem_ptr();
    Data* const host_mem_ptr2 = setup.get_host_input2_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max() / 10);
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr1,
        host_mem_ptr1 + size,
        [&distribution, &generator]() { return distribution(generator); });
    std::generate(
        host_mem_ptr2,
        host_mem_ptr2 + size,
        [&distribution, &generator]() { return distribution(generator); });

    MathOperator<Data> transform;

    setup.run(transform);

    std::vector<Data> expected_result;
    expected_result.resize(size);
    std::transform(host_mem_ptr1, host_mem_ptr1 + size, host_mem_ptr2, expected_result.begin(), transform);

    Data const* const result_ptr = setup.get_host_output_mem_ptr();
    std::vector<Data> result(result_ptr, result_ptr + size);

    REQUIRE(result == expected_result);
}


TEMPLATE_TEST_CASE(
    "Test transform double input with lambda and acc object",
    "[transform][lambda][Acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = std::uint64_t;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim, Data>(size)));

    vikunja::test::transform::TestSetupTransformDoubleInput<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr1 = setup.get_host_input1_mem_ptr();
    Data* const host_mem_ptr2 = setup.get_host_input2_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max());
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr1,
        host_mem_ptr1 + size,
        [&distribution, &generator]() { return distribution(generator); });
    std::generate(
        host_mem_ptr2,
        host_mem_ptr2 + size,
        [&distribution, &generator]() { return distribution(generator); });

    auto transform
        = [] ALPAKA_FN_HOST_ACC(alpaka::ExampleDefaultAcc<Dim, std::uint64_t> const& acc, Data const i, Data const j)
    { return alpaka::math::min(acc, i, j); };

    setup.run(transform);

    std::vector<Data> expected_result;
    expected_result.resize(size);
    std::transform(
        host_mem_ptr1,
        host_mem_ptr1 + size,
        host_mem_ptr2,
        expected_result.begin(),
        [](Data const i, Data const j) { return std::min(i, j); });

    Data const* const result_ptr = setup.get_host_output_mem_ptr();
    std::vector<Data> result(result_ptr, result_ptr + size);

    REQUIRE(result == expected_result);
}
