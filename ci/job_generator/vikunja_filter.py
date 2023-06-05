from alpaka_job_coverage.util import is_in_row
from alpaka_job_coverage.util import (
    row_check_name,
    row_check_version,
    row_check_backend_version,
)

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from vikunja_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


from packaging import version as pk_version


def vikunja_post_filter(row: List) -> bool:
    if is_in_row(row, ALPAKA):
        # the minimum boost version for alpaka 0.9.0 is 1.74.0
        if row_check_version(row, ALPAKA, ">", "0.8.0") and row_check_version(
            row, BOOST, "<", "1.74.0"
        ):
            return False

        # develop branch with commit 88860c9 supports
        # >= CMake 3.22
        # >= Clang 9
        # >= GCC 9
        if row_check_version(row, ALPAKA, ">", "0.9.0"):
            if row_check_version(row, CMAKE, "<", "3.22"):
                return False
            if row_check_name(row, HOST_COMPILER, "==", GCC) and row_check_version(
                row, HOST_COMPILER, "<", "9"
            ):
                return False
            if row_check_name(row, HOST_COMPILER, "==", CLANG) and row_check_version(
                row, HOST_COMPILER, "<", "9"
            ):
                return False
            if row_check_name(
                row, DEVICE_COMPILER, "==", CLANG_CUDA
            ) and row_check_version(row, DEVICE_COMPILER, "<", "9"):
                return False

    # CUDA 11.3+ is only supported by alpaka 0.7.0 an newer
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_version(row, DEVICE_COMPILER, ">=", "11.3")
        and row_check_version(row, ALPAKA, "<", "0.7.0")
    ):
        return False

    # CUDA 11.x and Clang 7 as host compiler does not support C++17
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_name(row, HOST_COMPILER, "==", CLANG)
        and row_check_version(row, HOST_COMPILER, "<", "8")
    ):
        return False

    # The combination of CUDA 11.3 - 11.5, Clang as host compiler and libstdc++ 9.4 is broken
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_name(row, HOST_COMPILER, "==", CLANG)
        and row_check_version(row, DEVICE_COMPILER, ">=", "11.3")
        and row_check_version(row, DEVICE_COMPILER, "<=", "11.5")
    ):
        return False

    # disable HIP for alpaka 0.8.0 and older, because of missing HIP 4.3+ support
    if row_check_name(row, DEVICE_COMPILER, "==", HIPCC) and row_check_version(
        row, ALPAKA, "<", "0.9.0"
    ):
        return False

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA):
        # alpaka 0.6.x does not support native CMake CUDA support and vikunja also not
        if row_check_version(row, ALPAKA, "<", "0.7.0"):
            return False

        # cmake 3.18 cannot compile simple example, if Clang is the CUDA compiler
        if row_check_version(row, CMAKE, "<", "3.19.0"):
            return False

        # Clang 11 has problems to detect the correct CUDA SDK version
        if row_check_version(row, DEVICE_COMPILER, "<=", "11"):
            return False

    return True
