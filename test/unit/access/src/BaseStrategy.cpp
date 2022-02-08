/* Copyright 2022 Hauke Mewes, Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/access/BaseStrategy.hpp>
#include <vikunja/access/PolicyBasedBlockStrategy.hpp>

#include <alpaka/alpaka.hpp>

#include <vector>

#include <catch2/catch.hpp>

using namespace vikunja::MemAccess;

TEST_CASE("BaseStrategy", "[MemAccessStrategy]")
{
    using Idx = std::size_t;
    constexpr auto size = static_cast<Idx>(64);


    BaseStrategy<Idx> zeroFirst(0, size);
    BaseStrategy<Idx> zeroSecond(0, size);
    BaseStrategy<Idx> one(1, size);

    SECTION("test comparison operators")
    {
        REQUIRE(zeroFirst == zeroSecond);
        REQUIRE(zeroSecond == zeroFirst);
        REQUIRE(zeroFirst != one);
        REQUIRE(one != zeroFirst);
        REQUIRE(zeroFirst < one);
        REQUIRE(one > zeroFirst);
        REQUIRE_FALSE(zeroFirst < zeroSecond);
        REQUIRE_FALSE(zeroFirst > zeroSecond);
        REQUIRE(zeroFirst <= zeroSecond);
        REQUIRE(zeroSecond <= zeroFirst);
        REQUIRE(zeroFirst <= one);
    }
    SECTION("test access operators")
    {
        REQUIRE(*zeroFirst == 0);
        REQUIRE(*one == 1);
        *zeroFirst = 2;
        REQUIRE(*zeroFirst == 2);
        REQUIRE(*zeroSecond == 0);
        REQUIRE_FALSE(*zeroFirst == *zeroSecond);
    }
    SECTION("test copy operator")
    {
        BaseStrategy<Idx> copyOfZeroFirst(zeroFirst);
        REQUIRE(copyOfZeroFirst == zeroFirst);
        *copyOfZeroFirst = 3;
        REQUIRE(copyOfZeroFirst != zeroFirst);
    }
}
