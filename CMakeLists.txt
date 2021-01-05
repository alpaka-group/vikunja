cmake_minimum_required(VERSION 3.10)
project(vikunja)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
option(VIKUNJA_REDUCE_COMPARING_BENCHMARKS_ENABLE "Should the reduce integration test run some comparison benchmarks?" OFF)
option(VIKUNJA_TRANSFORM_COMPARING_BENCHMARKS_ENABLE "Should the transform integration test run some comparison benchmarks?" OFF)

if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  list(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
  set(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED ON)
endif()
find_package(alpaka REQUIRED)

if(ALPAKA_ACC_GPU_CUDA_ENABLE)
    # activate support for host/device lambdas in cuda, currently (as of CUDA 10) still an experimental feature,
    # but it seems to work.
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda")
endif()


add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}
  INTERFACE
  alpaka::alpaka)
add_library(vikunja::vikunja ALIAS ${PROJECT_NAME})


if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  # Use doxygen for api doc when available.
  find_package(Doxygen QUIET)
  if(DOXYGEN_FOUND)
    doxygen_add_docs(doc include/)
  endif()

  enable_testing()
  add_subdirectory("test/")
  add_subdirectory("example/")
endif()
