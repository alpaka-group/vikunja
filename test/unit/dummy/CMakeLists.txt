cmake_minimum_required(VERSION 3.18)

set(_TARGET_NAME "test_dummy")


alpaka_add_executable(
        ${_TARGET_NAME}
        src/Dummy.cpp)
target_include_directories(
        ${_TARGET_NAME}
        PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE "vikunjaTestLib")

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_VIKUNJA_TEST_OPTIONS})
