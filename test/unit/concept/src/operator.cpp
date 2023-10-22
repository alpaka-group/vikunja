/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/concept/operator.hpp>

#include <alpaka/alpaka.hpp>

#include <utility>

#include <catch2/catch.hpp>

struct DummyAcc
{
    ALPAKA_FN_HOST_ACC int iMin(const int a, const int b) const
    {
        return (b < a) ? b : a;
    }

    ALPAKA_FN_HOST_ACC int iMax(const int a, const int b) const
    {
        return (a < b) ? b : a;
    }
};


template<
    typename TAcc,
    typename F,
    typename TData,
    typename TOperator = vikunja::concept::UnaryOp<TAcc, F, TData>,
    typename TRed = typename TOperator::TRed>
ALPAKA_FN_HOST_ACC auto unaryRunner(TAcc const& acc, F f, TData const arg) -> TRed
{
    return TOperator::run(acc, f, arg);
}


ALPAKA_FN_HOST_ACC int uFunc1(float const a)
{
    return static_cast<int>(a) + 2;
}
ALPAKA_FN_HOST_ACC auto uFunc2(float const a)
{
    return a * 3;
}
ALPAKA_FN_HOST_ACC float uFunc3(int const a)
{
    return 1.3f + static_cast<float>(a);
}
template<typename TRed, typename TData>
ALPAKA_FN_HOST_ACC TRed uFunc4(TData const a)
{
    return static_cast<TRed>(a);
}

template<typename TAcc>
ALPAKA_FN_HOST_ACC int uFunc5(TAcc const& acc, int const a)
{
    return acc.iMax(1, a);
}

template<typename TAcc, typename TData, int TMax>
ALPAKA_FN_HOST_ACC TData uFunc6(TAcc const& acc, TData const a)
{
    return acc.iMax(TMax, a);
}

struct UStruct1
{
    ALPAKA_FN_HOST_ACC int operator()(int const a) const
    {
        return 2 * a;
    }
};

template<typename TAcc, typename TData, typename TRet>
struct UStruct2
{
    ALPAKA_FN_HOST_ACC TRet operator()(TAcc const& acc, TData const a) const
    {
        return static_cast<TRet>(acc.iMax(static_cast<int>(a * 0.3), a));
    }
};

template<typename TAcc, typename TData>
struct UMakePair
{
    ALPAKA_FN_HOST_ACC std::pair<TData, TData> operator()(TAcc const& acc, TData const& a) const
    {
        return std::make_pair(a, a);
    }
};

TEST_CASE("UnaryOp", "[operators]")
{
    DummyAcc dummyAcc;

    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc1, 3.5f) == 5);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc2, 1.5f) == 4.5f);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc3, 7) == 8.3f);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc4<unsigned int, double>, 1.2) == 1);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc5<DummyAcc>, 2) == 2);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uFunc6<DummyAcc, int, 7>, 3) == 7);

    auto uLambda1 = [] ALPAKA_FN_HOST_ACC(float const a) { return static_cast<unsigned int>(a); };
    auto uLambda2 = [](auto const a) { return a && true; };
    auto uLambda3 = [](auto const&, int const a) { return 2 * a; };
    auto uLambda4 = [](auto const&, auto const a) -> int { return 7.5f - a; };
    auto uLambda5 = [](auto const&, auto const a) { return 7.5f - a; };
    auto uLambda6 = [] ALPAKA_FN_HOST_ACC(DummyAcc const& acc, int const a) { return acc.iMin(a, 0); };

    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda1, 3.5) == 3);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda2, false) == false);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda3, 2) == 4);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda4, 3.f) == 4);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda5, 3.f) == 4.5f);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uLambda6, -3) == -3);

    UStruct1 uStruct1;
    UStruct2<DummyAcc, double, int> uStruct2;
    UMakePair<DummyAcc, double> makePair;
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uStruct1, 1) == 2);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, uStruct2, 6.8) == 6);
    REQUIRE(unaryRunner<DummyAcc>(dummyAcc, makePair, 1.2) == std::make_pair(1.2, 1.2));
}


template<
    typename TAcc,
    typename F,
    typename TData1,
    typename TData2,
    typename TOperator = vikunja::concept::BinaryOp<TAcc, F, TData1, TData2>,
    typename TRed = typename TOperator::TRed>
ALPAKA_FN_HOST_ACC auto binaryRunner(TAcc const& acc, F f, TData1 const arg1, TData2 const arg2) -> TRed
{
    return TOperator::run(acc, f, arg1, arg2);
}


ALPAKA_FN_HOST_ACC int bFunc1(int const a, int const b)
{
    return a + b;
}

ALPAKA_FN_HOST_ACC float bFunc2(int const a, int const b)
{
    return static_cast<float>(a + b) + 0.5f;
}

template<typename TAcc>
ALPAKA_FN_HOST_ACC int bFunc3(TAcc const&, unsigned int const a, double const b)
{
    return (b * a) - 0.5;
}


template<typename TAcc>
ALPAKA_FN_HOST_ACC int bFunc4(TAcc const&, int const a, int const b)
{
    return (a * b) * 2;
}

template<typename TAcc, typename TData>
ALPAKA_FN_HOST_ACC int bFunc5(TAcc const&, TData const a, TData const b)
{
    return (a * b) * 2;
}

template<typename TAcc, typename TRed, typename TData1, typename TData2>
ALPAKA_FN_HOST_ACC TRed bFunc6(TAcc const& acc, TData1 const a, TData2 const b)
{
    return static_cast<TRed>(acc.iMax(static_cast<int>(a), 0) * b);
}


struct BStruct1
{
    ALPAKA_FN_HOST_ACC int operator()(float const a, double const b) const
    {
        return static_cast<int>(static_cast<double>(a) * b);
    }
};

template<typename TAcc, typename TData1, typename TData2, typename TRet>
struct BStruct2
{
    ALPAKA_FN_HOST_ACC TRet operator()(TAcc const& acc, TData1 const a, TData2 const b) const
    {
        return static_cast<TRet>(acc.iMin(static_cast<int>(a), static_cast<int>(b)));
    }
};

template<typename TAcc, typename TData1, typename TData2>
struct BMakePair
{
    ALPAKA_FN_HOST_ACC std::pair<TData1, TData2> operator()(TAcc const& acc, TData1 const& a, TData2 const& b) const
    {
        return std::make_pair(a, b);
    }
};

TEST_CASE("BinaryOp", "[operators]")
{
    DummyAcc dummyAcc;

    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc1, 1, 2) == 3);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc2, 4, 6) == 10.5f);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc3<DummyAcc>, 7, 3.0) == 20);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc4<DummyAcc>, 1, 1) == 2);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc5<DummyAcc, int>, 4, 8) == 64);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bFunc6<DummyAcc, int, double, float>, -3, 1245.f) == 0.f);

    auto bLambda1
        = [] ALPAKA_FN_HOST_ACC(double const a, double const b) { return static_cast<unsigned int>((a * 2.5) / b); };
    auto bLambda2
        = [](auto const&, double const a, double const b) { return static_cast<unsigned int>((a * 2.5) / b); };
    auto bLambda3 = [] ALPAKA_FN_HOST_ACC(DummyAcc const& acc, float const a, double const b) -> int
    { return acc.iMin(static_cast<int>(a * b), 10); };

    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bLambda1, 3.0, 2.0) == 3);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bLambda2, 3.0, 5) == 1);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bLambda3, 1.2f, 3.2) == 3);

    BStruct1 bStruct1;
    BStruct2<DummyAcc, int, float, double> bStruct2;
    BMakePair<DummyAcc, int, float> makePair;
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bStruct1, 1.3f, 4.7) == 6);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, bStruct2, 3, 1.2) == 1.0);
    REQUIRE(binaryRunner<DummyAcc>(dummyAcc, makePair, 1, 3.4f) == std::make_pair(1, 3.4f));
}
