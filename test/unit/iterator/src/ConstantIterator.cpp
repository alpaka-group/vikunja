/* Copyright 2021 Anton Reinhard
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/mem/iterator/ConstantIterator.hpp>

#include <alpaka/alpaka.hpp>

#include <catch2/catch.hpp>

using vikunja::mem::iterator::ConstantIterator;

TEMPLATE_TEST_CASE("Test Constant Iterator", "", int, double, uint64_t)
{
    using Type = TestType;
    using Idx = int64_t;

    GIVEN("A default ConstantIterator")
    {
        ConstantIterator<Type> a(1);
        THEN("its index is zero")
        {
            REQUIRE(a == ConstantIterator<Type>(1, 0));
            REQUIRE(a + 0 == a);
        }
    }

    GIVEN("A ConstantIterator")
    {
        const Type value = GENERATE(take(5, random(-50, 50)));
        const Idx idx = GENERATE(take(3, random(-20, 20)));
        ConstantIterator<Type> a(value, idx);

        THEN("the iterator has correct values set")
        {
            REQUIRE(a + 0 == a);
            REQUIRE(*a == value);
            REQUIRE(a[5] == value);
            REQUIRE(a[-5] == value);

            REQUIRE(a == a);
            REQUIRE(a <= a);
            REQUIRE(a >= a);
            REQUIRE_FALSE(a != a);
            REQUIRE_FALSE(a < a);
            REQUIRE_FALSE(a > a);
        }

        WHEN("creating a second iterator from this one")
        {
            // use a positive offset so we can make more assumptions
            const Idx offset = GENERATE(take(3, random(1, 20)));
            auto b = a + offset;
            THEN("comparisons are correct")
            {
                REQUIRE(a < b);
                REQUIRE(a <= b);
                REQUIRE(a != b);
                REQUIRE_FALSE(a > b);
                REQUIRE_FALSE(a >= b);
                REQUIRE_FALSE(a == b);
            }

            THEN("new iterator has the correct value set")
            {
                REQUIRE(*b == value);
                REQUIRE(b[5] == value);
                REQUIRE(b[-5] == value);
            }

            THEN("arithmetic operators on the iterators work correctly")
            {
                auto c = a + idx + offset;
                REQUIRE(c - a == idx + offset);
                REQUIRE(*c == value);

                auto d = b - offset;
                REQUIRE(d == a);
                REQUIRE(d < b);
                REQUIRE(*d == value);
            }
        }

        THEN("muting operators work correctly")
        {
            REQUIRE_NOTHROW(a += 5);
            REQUIRE(*a == value);
            REQUIRE(a == a);
            REQUIRE(a[5] == value);

            REQUIRE_NOTHROW(a -= 10);
            REQUIRE(*a == value);
            REQUIRE(a == a);
            REQUIRE(a[5] == value);
        }
    }
}
