/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/operators/operators.hpp>
#include <alpaka/alpaka.hpp>
#include <catch2/catch.hpp>

struct DummyAcc
{
    ALPAKA_FN_HOST_ACC int iMin(const int& a, const int& b) const
    {
        return (b < a) ? b : a;
    }

    ALPAKA_FN_HOST_ACC int iMax(const int& a, const int& b) const
    {
        return (a < b) ? b : a;
    }
};


template<
    typename TAcc,
    typename F,
    typename TData,
    typename TOperator = vikunja::operators::UnaryOp<TAcc, F, TData>,
    typename TRed = typename TOperator::TRed>
auto unaryRunner(TAcc const& acc, F f, TData arg) -> TRed
{
    return TOperator::run(acc, f, arg);
}


int uFunc1(float a)
{
    return static_cast<int>(a) + 2;
}
auto uFunc2(float a)
{
    return a * 3;
}
float ALPAKA_FN_HOST_ACC uFunc3(int a)
{
    return 1.3f + static_cast<float>(a);
}
template<typename TRed, typename TData>
TRed ALPAKA_FN_HOST_ACC uFunc4(TData a)
{
    return static_cast<TRed>(a);
}

template<typename TAcc>
int uFunc5(TAcc const& acc, int a)
{
    return acc.iMax(1, a);
}

template<typename TAcc, typename TData, int TMax>
TData ALPAKA_FN_HOST_ACC uFunc6(TAcc const& acc, TData a)
{
    return acc.iMax(TMax, a);
}


TEST_CASE("UnaryOp", "[operators]")
{
    DummyAcc dummyAcc;

    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc1, 3.5f) == 5);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc2, 1.5f) == 4.5f);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc3, 7) == 8.3f);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc4<unsigned int, double>, 1.2) == 1);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc5<DummyAcc>, 2) == 2);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc6<DummyAcc, int, 7>, 3) == 7);

    auto uLambda1 = [] ALPAKA_FN_HOST_ACC(float a) { return static_cast<unsigned int>(a); };
    auto uLambda2 = [](auto a) { return a && true; };
    auto uLambda3 = [](auto const&, int a) { return 2 * a; };
    auto uLambda4 = [](auto const&, auto a) -> int { return 7.5f - a; };
    auto uLambda5 = [](auto const&, auto a) { return 7.5f - a; };
    auto uLambda6 = [] ALPAKA_FN_HOST_ACC(DummyAcc const& acc, int a) { return acc.iMin(a, 0); };

    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda1, 3.5) == 3);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda2, false) == false);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda3, 2) == 4);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda4, 3.f) == 4);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda5, 3.f) == 4.5f);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda6, -3) == -3);
}


template<
    typename TAcc,
    typename F,
    typename TData1,
    typename TData2,
    typename TOperator = vikunja::operators::BinaryOp<TAcc, F, TData1, TData2>,
    typename TRed = typename TOperator::TRed>
auto binaryRunner(TAcc const& acc, F f, TData1 arg1, TData2 arg2) -> TRed
{
    return TOperator::run(acc, f, arg1, arg2);
}


int bFunc1(int a, int b)
{
    return a + b;
}

ALPAKA_FN_HOST_ACC float bFunc2(int a, int b)
{
    return static_cast<float>(a + b) + 0.5f;
}

template<typename TAcc>
ALPAKA_FN_HOST_ACC int bFunc3(TAcc const&, unsigned int a, double b)
{
    return (b * a) - 0.5;
}


template<typename TAcc>
ALPAKA_FN_HOST_ACC int bFunc4(TAcc const&, int a, int b)
{
    return (a * b) * 2;
}

template<typename TAcc, typename TData>
ALPAKA_FN_HOST_ACC int bFunc5(TAcc const&, TData a, TData b)
{
    return (a * b) * 2;
}

template<typename TAcc, typename TRed, typename TData1, typename TData2>
ALPAKA_FN_HOST_ACC TRed bFunc6(TAcc const& acc, TData1 a, TData2 b)
{
    return static_cast<TRed>(acc.iMax(static_cast<int>(a), 0) * b);
}


TEST_CASE("BinaryOp", "[operators]")
{
    DummyAcc dummyAcc;

    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc1, 1, 2) == 3);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc2, 4, 6) == 10.5f);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc3<DummyAcc>, 7, 3.0) == 20);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc4<DummyAcc>, 1, 1) == 2);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc5<DummyAcc, int>, 4, 8) == 64);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc6<DummyAcc, int, double, float>, -3, 1245.f) == 0.f);

    auto bLambda1 = [] ALPAKA_FN_HOST_ACC(double a, double b) { return static_cast<unsigned int>((a * 2.5) / b); };
    auto bLambda2 = [](auto const&, double a, double b) { return static_cast<unsigned int>((a * 2.5) / b); };
    auto bLambda3 = [] ALPAKA_FN_HOST_ACC(DummyAcc const& acc, float a, double b) -> int {
        return acc.iMin(static_cast<int>(a * b), 10);
    };

    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bLambda1, 3.0, 2.0) == 3);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bLambda2, 3.0, 5) == 1);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bLambda3, 1.2f, 3.2) == 3);
}
