cmake_minimum_required(VERSION 3.10)
project(vikunja)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

list(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
find_package(alpaka REQUIRED)

add_subdirectory(src)
# Use doxygen for api doc when available.
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    doxygen_add_docs(doc src/ include/)
endif()