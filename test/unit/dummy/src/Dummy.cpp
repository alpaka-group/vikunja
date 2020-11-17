#include <catch2/catch.hpp>

int two()
{
    return 2;
}

TEST_CASE("DummyTest", "[dummy]")
{
    REQUIRE(two() == 2);
}