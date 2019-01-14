//
// Created by hauke on 11.01.19.
//

#include <vikunja/test/AlpakaSetup.hpp>
#include <vikunja/reduce/detail/BlockThreadReduceKernel.hpp>
#include <alpaka/alpaka.hpp>
#include <catch2/catch.hpp>
#include <cstdlib>
#include <alpaka/acc/AccCpuSerial.hpp>

TEST_CASE("Test policies", "[reduce]")
{
    using namespace vikunja::reduce::detail;

    using TestAlpakaEnv = vikunja::test::TestAlpakaSetup<alpaka::dim::DimInt<1u>, uint64_t, alpaka::acc::AccCpuSerial, alpaka::acc::AccCpuSerial, alpaka::queue::QueueCpuSync>;


    SECTION("Test LinearMemAccessPolicy") {
    }
}