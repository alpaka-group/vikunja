cmake_minimum_required(VERSION 3.10)
project(vikunja
  VERSION      0.1.0
  DESCRIPTION  "Primitives for Alpaka"
  HOMEPAGE_URL "https://github.com/ComputationalRadiationPhysics/vikunja"
  LANGUAGES    CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
option(VIKUNJA_REDUCE_COMPARING_BENCHMARKS_ENABLE "Should the reduce integration test run some comparison benchmarks?" OFF)
option(VIKUNJA_TRANSFORM_COMPARING_BENCHMARKS_ENABLE "Should the transform integration test run some comparison benchmarks?" OFF)

if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  list(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
  option(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE "enable alpaka serial accelerator" ON)
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
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}
  INTERFACE
  alpaka::alpaka)
add_library(vikunja::vikunja ALIAS ${PROJECT_NAME})


if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  #########################################
  ######### Install Configuration #########
  #########################################

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION 0.1
    COMPATIBILITY AnyNewerVersion
    )

  include(GNUInstallDirs)
  install(TARGETS ${PROJECT_NAME} alpaka
    EXPORT "${PROJECT_NAME}Targets"
    )

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )

  install(EXPORT "${PROJECT_NAME}Targets"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    NAMESPACE ${PROJECT_NAME}::)
  install(FILES
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

  #########################################
  ############# Documentation #############
  #########################################

  # Use doxygen for api doc when available.
  find_package(Doxygen QUIET)
  if(DOXYGEN_FOUND)
    doxygen_add_docs(doc include/)
  endif()

  #########################################
  ################ Testing ################
  #########################################

  enable_testing()
  add_subdirectory("test/")
  add_subdirectory("example/")
endif()
