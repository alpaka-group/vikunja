cmake_minimum_required(VERSION 3.10)
set(_TARGET_NAME "example_transform")
alpaka_add_executable(${_TARGET_NAME} src/transform-main.cpp)
target_link_libraries(${_TARGET_NAME} PUBLIC vikunja::internalvikunja)
