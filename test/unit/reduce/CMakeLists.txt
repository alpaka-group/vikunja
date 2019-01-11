cmake_minimum_required(VERSION 3.10)

set(_TARGET_NAME "reduce")


alpaka_add_executable(
        ${_TARGET_NAME}
        Reduce.cpp)
target_include_directories(
        ${_TARGET_NAME}
        PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE "CatchMain")

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_ALPAKA_TEST_OPTIONS})