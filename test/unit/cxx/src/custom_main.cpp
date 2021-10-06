/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include "alpaka_cxx.hpp"

int ALPAKA_CXX = 0;

int main(int argc, char* argv[])
{
    Catch::Session session;

    using namespace Catch::clara;
    auto cli = session.cli() // Get Catch's composite command line parser
        | Opt(ALPAKA_CXX,
              "C++ standard set for alpaka and vikunja") // bind variable to a new option, with a hint string
            ["--cxx"] // the option names it will respond to
        ("Set C++ standard"); // description string for the help output

    session.cli(cli);

    int returnCode = session.applyCommandLine(argc, argv);
    if(returnCode != 0)
        return returnCode;

    return session.run();
}
