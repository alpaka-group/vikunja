cmake_minimum_required(VERSION 3.10)

set(_TARGET_NAME "reduceTest")


alpaka_add_executable(
        ${_TARGET_NAME}
        src/Reduce.cpp)
target_include_directories(
        ${_TARGET_NAME}
        PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(${_TARGET_NAME} PRIVATE ${alpaka_INCLUDE_DIRS})
target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE "vikunjaTestLib")

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")
if(VIKUNJA_REDUCE_COMPARING_BENCHMARKS_ENABLE)
    target_compile_definitions(${_TARGET_NAME} PRIVATE "VIKUNJA_REDUCE_COMPARING_BENCHMARKS")
endif()

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_VIKUNJA_TEST_OPTIONS})