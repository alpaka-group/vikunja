//
// Created by hauke on 10.01.19.
//
#include <catch2/catch.hpp>

int two() {
    return 2;
}

TEST_CASE("DummyTest", "[dummy]")
{
    REQUIRE(two() == 2);
}