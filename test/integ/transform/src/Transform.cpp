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
                typename TDataResult = TData,
                typename TIdx = std::uint64_t>
            class TestSetupTransform : public TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>
            {
            public:
                using TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>::TestSetupBase;

                using Base = typename vikunja::test::transform::TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>;

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
                }
            };

            template<
                typename TDim,
                template<typename, typename>
                class TAcc,
                typename TData,
                typename TDataResult = TData,
                typename TIdx = std::uint64_t>
            class TestSetupTransformPtr : public TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>
            {
            public:
                using TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>::TestSetupBase;

                using Base = typename vikunja::test::transform::TestSetupBase<TDim, TAcc, TData, TDataResult, TIdx>;

                template<typename TReduceFunctor>
                void run(TReduceFunctor reduceFunctor)
                {
                    alpaka::memcpy(
                        Base::Base::queueAcc,
                        Base::m_device_input1_mem,
                        Base::m_host_input1_mem,
                        Base::m_extent);

                    TData* begin = alpaka::getPtrNative(Base::m_device_input1_mem);
                    TData* end = begin + Base::m_size;

                    vikunja::transform::deviceTransform<typename Base::Acc>(
                        Base::devAcc,
                        Base::Base::queueAcc,
                        begin,
                        end,
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
                typename TData1,
                typename TData2 = TData1,
                typename TDataResult = TData1,
                typename TIdx = std::uint64_t>
            class TestSetupTransformDoubleInput : public TestSetupBase<TDim, TAcc, TData1, TDataResult, TIdx>
            {
            private:
                using Base = typename vikunja::test::transform::TestSetupBase<TDim, TAcc, TData1, TDataResult, TIdx>;
                using BufHost = alpaka::Buf<typename Base::Base::Host, TData2, TDim, TIdx>;
                using BufDev = alpaka::Buf<typename Base::Base::Acc, TData2, TDim, TIdx>;

                BufHost m_host_input2_mem;
                BufDev m_device_input2_mem;

            public:
                TestSetupTransformDoubleInput(uint64_t const memSize)
                    : TestSetupBase<TDim, TAcc, TData1, TDataResult, TIdx>(memSize)
                    , m_host_input2_mem(alpaka::allocBuf<TData2, TIdx>(Base::Base::devHost, Base::m_extent))
                    , m_device_input2_mem(alpaka::allocBuf<TData2, TIdx>(Base::Base::devAcc, Base::m_extent))
                {
                }


                TData2* get_host_input2_mem_ptr()
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
                }
            };

            template<
                typename TDim,
                template<typename, typename>
                class TAcc,
                typename TData1,
                typename TData2 = TData1,
                typename TDataResult = TData1,
                typename TIdx = std::uint64_t>
            class TestSetupTransformDoubleInputPtr : public TestSetupBase<TDim, TAcc, TData1, TDataResult, TIdx>
            {
            private:
                using Base = typename vikunja::test::transform::TestSetupBase<TDim, TAcc, TData1, TDataResult, TIdx>;
                using BufHost = alpaka::Buf<typename Base::Base::Host, TData2, TDim, TIdx>;
                using BufDev = alpaka::Buf<typename Base::Base::Acc, TData2, TDim, TIdx>;

                BufHost m_host_input2_mem;
                BufDev m_device_input2_mem;

            public:
                TestSetupTransformDoubleInputPtr(uint64_t const memSize)
                    : TestSetupBase<TDim, TAcc, TData1, TDataResult, TIdx>(memSize)
                    , m_host_input2_mem(alpaka::allocBuf<TData2, TIdx>(Base::Base::devHost, Base::m_extent))
                    , m_device_input2_mem(alpaka::allocBuf<TData2, TIdx>(Base::Base::devAcc, Base::m_extent))
                {
                }


                TData2* get_host_input2_mem_ptr()
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

                    TData1* begin = alpaka::getPtrNative(Base::m_device_input1_mem);
                    TData1* end = begin + Base::m_size;

                    vikunja::transform::deviceTransform<typename Base::Acc>(
                        Base::devAcc,
                        Base::Base::queueAcc,
                        begin,
                        end,
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


struct IncOne
{
    template<typename TData>
    ALPAKA_FN_HOST_ACC TData operator()(TData const val) const
    {
        return val + 1;
    }
};

template<int TMaxValue>
struct Max
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_HOST_ACC TData operator()(TAcc const& acc, TData const i) const
    {
        return alpaka::math::max(acc, i, TMaxValue);
    }
};

struct MathOperator
{
    template<typename TData>
    ALPAKA_FN_HOST_ACC TData operator()(TData const i, TData const j) const
    {
        return ((i * 2) - (i + 1)) + ((j * 2) - (j + 1));
    }
};

template<typename TReturn, typename TData1, typename TData2>
struct Sub
{
    ALPAKA_FN_HOST_ACC TReturn operator()(TData1 const i, TData2 const j) const
    {
        return static_cast<TReturn>(i) - static_cast<TReturn>(j);
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

    INFO((vikunja::test::print_acc_info<Dim>(size)));

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

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::transform::TestSetupTransformPtr<Dim, alpaka::ExampleDefaultAcc, Data> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_input1_mem_ptr();
    std::iota(host_mem_ptr, host_mem_ptr + size, 1);

    IncOne transform;

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

    INFO((vikunja::test::print_acc_info<Dim>(size)));

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

    INFO((vikunja::test::print_acc_info<Dim>(size)));

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

    Max<(1 << 10)> transform;

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
    "Test transform with lamba, acc object and different types for input and output",
    "[transform][lambda][Acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data = int;
    using ReturnType = float;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::transform::TestSetupTransform<Dim, alpaka::ExampleDefaultAcc, Data, ReturnType> setup(size);

    // setup initial values
    Data* const host_mem_ptr = setup.get_host_input1_mem_ptr();
    std::uniform_int_distribution<Data> distribution(
        std::numeric_limits<Data>::min(),
        std::numeric_limits<Data>::max() - 1);
    std::default_random_engine generator;
    std::generate(
        host_mem_ptr,
        host_mem_ptr + size,
        [&distribution, &generator]() { return distribution(generator); });

    auto transform
        = [] ALPAKA_FN_HOST_ACC(alpaka::ExampleDefaultAcc<Dim, std::uint64_t> const& acc, Data const i) -> ReturnType
    { return alpaka::math::max(acc, static_cast<ReturnType>(i), static_cast<ReturnType>(1 << 10)) + 0.5f; };

    setup.run(transform);

    std::vector<ReturnType> expected_result;
    expected_result.resize(size);
    std::transform(
        host_mem_ptr,
        host_mem_ptr + size,
        expected_result.begin(),
        [](Data const i) -> ReturnType
        { return std::max(static_cast<ReturnType>(i), static_cast<ReturnType>(1 << 10)) + 0.5f; });

    ReturnType const* const result_ptr = setup.get_host_output_mem_ptr();
    std::vector<ReturnType> result(result_ptr, result_ptr + size);

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

    INFO((vikunja::test::print_acc_info<Dim>(size)));

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

    MathOperator transform;

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

    INFO((vikunja::test::print_acc_info<Dim>(size)));

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

TEMPLATE_TEST_CASE(
    "Test transform double input with operator, without acc object and different types for input1, input2 and return "
    "value",
    "[transform][operator][Acc]",
    (alpaka::DimInt<1u>),
    (alpaka::DimInt<2u>),
    (alpaka::DimInt<3u>) )
{
    using Dim = TestType;
    using Data1 = std::uint64_t;
    using Data2 = float;
    using ReturnType = int;
    int const min_value = 0;
    int const max_value = 1000;

    auto size = GENERATE(1, 10, 777, 1 << 10);

    INFO((vikunja::test::print_acc_info<Dim>(size)));

    vikunja::test::transform::
        TestSetupTransformDoubleInputPtr<Dim, alpaka::ExampleDefaultAcc, Data1, Data2, ReturnType>
            setup(size);

    // setup initial values
    Data1* const host_mem_ptr1 = setup.get_host_input1_mem_ptr();
    std::uniform_int_distribution<Data1> distribution1(static_cast<Data1>(min_value), static_cast<Data1>(max_value));
    std::default_random_engine generator1;
    std::generate(
        host_mem_ptr1,
        host_mem_ptr1 + size,
        [&distribution1, &generator1]() { return distribution1(generator1); });

    Data2* const host_mem_ptr2 = setup.get_host_input2_mem_ptr();
    std::uniform_real_distribution<Data2> distribution2(static_cast<Data2>(min_value), static_cast<Data2>(max_value));
    std::default_random_engine generator2;
    std::generate(
        host_mem_ptr2,
        host_mem_ptr2 + size,
        [&distribution2, &generator2]() { return distribution2(generator2); });

    Sub<ReturnType, Data1, Data2> transform;

    setup.run(transform);

    std::vector<ReturnType> expected_result;
    expected_result.resize(size);
    std::transform(host_mem_ptr1, host_mem_ptr1 + size, host_mem_ptr2, expected_result.begin(), transform);

    ReturnType const* const result_ptr = setup.get_host_output_mem_ptr();
    std::vector<ReturnType> result(result_ptr, result_ptr + size);

    REQUIRE(result == expected_result);
}
