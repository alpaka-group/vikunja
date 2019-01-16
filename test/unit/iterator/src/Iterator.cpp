//
// Created by mewes30 on 16.01.19.
//

#include <vikunja/mem/iterator/BaseIterator.hpp>
#include <alpaka/alpaka.hpp>
#include <catch2/catch.hpp>
#include <vector>

using Type = uint64_t;
using Idx = std::size_t;
using namespace vikunja::mem::iterator;

std::vector<Type> generateIndexVector(Idx size) {
    std::vector<Type> data(size);
    for(Idx i = 0; i < size; ++i) {
        data[i] = i;
    }
    return data;
}


TEST_CASE("BaseIterator", "[iterator]") {

    constexpr Idx size = 64;
    std::vector<Type> testData{generateIndexVector(size)};

    BaseIterator<Type> zeroFirst(testData.data(), 0, size);
    BaseIterator<Type> zeroSecond(testData.data(), 0, size);
    BaseIterator<Type> one(testData.data(), 1, size);
    BaseIterator<Type> copyOfZeroFirst(zeroFirst);


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
    REQUIRE(*zeroFirst == 0);
    REQUIRE(*one == 1);
    REQUIRE(copyOfZeroFirst == zeroFirst);

}