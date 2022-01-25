/* Copyright 2022 Anton Reinhard
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/mem/iterator/TransformIterator.hpp>

#include <alpaka/alpaka.hpp>

#include <iterator>
#include <vector>

#include <catch2/catch.hpp>

using Type = std::int64_t;
using IType = Type*;
using Idx = std::int64_t;
using namespace vikunja::mem::iterator;

inline Type increment(const Type& i)
{
    return i + 1;
}

inline Type twice(const Type& i)
{
    return 2 * i;
}

inline Type setZero(const Type& i)
{
    return 0;
}

TEST_CASE("transform_iterator", "[transform][iterator]")
{
    Idx size = GENERATE(0, 1, 50, 1000);
    Type value = GENERATE(-50, 0, 50);
    std::function func = GENERATE(increment, twice, setZero);

    std::vector<Type> vec;
    vec.resize(size, value);

    using ItType = std::vector<Type>::iterator;
    using Func = decltype(func);

    using TransformIt = vikunja::mem::iterator::TransformIterator<void*, Type, ItType, Func>;

    TransformIt tItBegin(vec.begin(), func);
    TransformIt tItEnd(vec.end(), func);

    while(tItBegin != tItEnd)
    {
        CHECK(*tItBegin == func(value));
        ++tItBegin;
    }

    REQUIRE(tItBegin == tItEnd);

    // .end() points *past* the last element, so start at 1
    for(Idx i = 1; i <= size; ++i)
    {
        CHECK(tItEnd[-i] == func(value));
        --tItBegin;
        CHECK(*tItBegin == func(value));
    }
}
