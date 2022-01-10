cmake_minimum_required(VERSION 3.18)
set(_TARGET_NAME "example_constant_iterator")
alpaka_add_executable(${_TARGET_NAME} src/constantIterator-main.cpp)
target_link_libraries(${_TARGET_NAME} PUBLIC vikunja::internalvikunja)
