/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <catch2/catch.hpp>
#include "alpaka_cxx.hpp"

TEST_CASE("TestCxxStandard", "[c++ standard]")
{
#if __cplusplus == 202002L
    int const compiled_cxx_version = 20;
#elif __cplusplus == 201703L
    int const compiled_cxx_version = 17;
#elif __cplusplus == 201402L
    int const compiled_cxx_version = 14;
#else
    // c++ version not supported by vikunja
    int const compiled_cxx_version = 0;
#endif

    INFO("compiled_cxx_version is set by the compiler");
    INFO("ALPAKA_CXX is passed as terminal argument via CMake");
    REQUIRE(compiled_cxx_version == ALPAKA_CXX);
}
