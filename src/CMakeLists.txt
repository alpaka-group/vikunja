cmake_minimum_required(VERSION 3.10)
if(alpaka_FOUND)
    # Reduce test
    alpaka_add_executable(reduceTest reduce.cpp)
    target_include_directories(reduceTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_link_libraries(reduceTest PUBLIC "alpaka")
    target_compile_options(reduceTest PUBLIC -fopt-info-vec-optimized)
endif()