cmake_minimum_required(VERSION 3.10)
if(alpaka_FOUND)
    # Reduce test
    alpaka_add_executable(reduceFiddle reduce.cpp)
    target_include_directories(reduceFiddle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_link_libraries(reduceFiddle PUBLIC "alpaka")
    target_compile_options(reduceFiddle PUBLIC -fopt-info-vec-optimized)
endif()