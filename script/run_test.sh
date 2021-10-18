#!/bin/bash

echo "container version -> $CONTAINER_VERSION"

: ${VIKUNJA_CXX?"VIKUNJA_CXX must be specified"}
: ${VIKUNJA_BOOST_VERSIONS?"VIKUNJA_BOOST_VERSIONS must be specified"}
: ${ALPAKA_ACCS?"ALPAKA_ACCS must be specified"}

# the default build type is Release
# if neccesary, you can rerun the pipeline with another build type-> https://docs.gitlab.com/ee/ci/pipelines.html#manually-executing-pipelines
# to change the build type, you must set the environment variable VIKUNJA_BUILD_TYPE

if [[ ! -v VIKUNJA_BUILD_TYPE ]] ; then
    VIKUNJA_BUILD_TYPE=Release ;
fi

###################################################
# cmake config builder
###################################################

VIKUNJA_CONST_ARGS="-DBUILD_TESTING=ON -DVIKUNJA_SYSTEM_CATCH2=OFF -DVIKUNJA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=${VIKUNJA_BUILD_TYPE}"

VIKUNJA_CMAKE_ARGS="${VIKUNJA_CMAKE_ARGS} ${VIKUNJA_CONST_ARGS}"

# if ALPAKA_CXX_STANDARD is defined in the job add it to the CMake arguments
if [[ -v ALPAKA_CXX_STANDARD ]] ; then
    VIKUNJA_CMAKE_ARGS="${VIKUNJA_CMAKE_ARGS} -DALPAKA_CXX_STANDARD=${ALPAKA_CXX_STANDARD} ";
fi

if [[ -v VIKUNJA_CXX_TEST ]] ; then
    VIKUNJA_CMAKE_ARGS="${VIKUNJA_CMAKE_ARGS} -DVIKUNJA_ENABLE_CXX_TEST=ON ";
fi

CMAKE_CONFIGS=()
for CXX_VERSION in $VIKUNJA_CXX; do
    for BOOST_VERSION in ${VIKUNJA_BOOST_VERSIONS}; do
	for ACC in ${ALPAKA_ACCS}; do
	    CONFIG_STRING="${VIKUNJA_CMAKE_ARGS} -DCMAKE_CXX_COMPILER=${CXX_VERSION} -DBOOST_ROOT=/opt/boost/${BOOST_VERSION} -D${ACC}=ON"

	    if [ "$ACC" = "ALPAKA_ACC_GPU_CUDA_ENABLE" ]; then
		CONFIG_STRING="${CONFIG_STRING} -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_HOST_COMPILER=${CXX_VERSION}"
	    fi

	    CMAKE_CONFIGS+=("${CONFIG_STRING}")
	done
    done
done

###################################################
# build an run tests
###################################################

base_dir=$(pwd)

for ALPAKA_VERSION in ${VIKUNJA_ALPAKA_VERSIONS}; do
    cd $base_dir
    # install alpaka
    git clone --depth 1 --branch ${ALPAKA_VERSION} https://github.com/alpaka-group/alpaka.git
    mkdir alpaka/build && cd alpaka/build
    cmake .. -DBOOST_ROOT=/opt/boost/1.73.0
    cmake --install .
    cd ../..
    rm -rf alpaka

    # use one build directory for all build configurations
    mkdir -p build
    cd build

    # ALPAKA_ACCS contains the backends, which are used for each build
    # the backends are set in the sepcialized base jobs .base_gcc,.base_clang and.base_cuda
    for CONFIG in $(seq 0 $((${#CMAKE_CONFIGS[*]} - 1))); do
	CMAKE_ARGS=${CMAKE_CONFIGS[$CONFIG]}
	echo -e "\033[0;32m///////////////////////////////////////////////////"
	echo "number of processor threads -> $(nproc)"
	echo "ALPAKA_VERSION -> ${ALPAKA_VERSION}"
	cmake --version | head -n 1
	echo "CMAKE_ARGS -> ${CMAKE_ARGS}"
	echo -e "/////////////////////////////////////////////////// \033[0m \n\n"

	cmake .. $CMAKE_ARGS
	cmake --build . -j
	ctest --output-on-failure

	# if ALPAKA_CXX_STANDARD is set manually, run this without ctest
	# to eliminate errors in CMake
	if [[ -v VIKUNJA_CXX_TEST ]] ; then
	    echo -e "test/unit/cxx/test_cxx --cxx ${ALPAKA_CXX_STANDARD}"
	    test/unit/cxx/test_cxx --cxx ${ALPAKA_CXX_STANDARD}
	fi

	rm -r *
    done

    # uninstall alpaka
    rm -r /usr/local/include/alpaka
    rm -r /usr/local/lib/cmake/alpaka
done
