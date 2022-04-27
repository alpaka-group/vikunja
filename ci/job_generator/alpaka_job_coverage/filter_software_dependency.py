"""Filter rules handling software dependencies and compiler settings.
"""

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import (
    row_check_backend_version,
    row_check_name,
    row_check_version,
    is_in_row,
)

from packaging import version as pk_version


def software_dependency_filter(row: List) -> bool:
    """Filter rules handling software dependencies and compiler settings.

    Args:
        row (List): Combination to verify. The row can contain
        up to all combination fields and at least two items.

    Returns:
        bool: True, if combination is valid, otherwise False.
    """

    # GCC 6 and below is not available on Ubuntu 20.04
    if row_check_version(row, UBUNTU, "==", "20.04"):
        if (
            row_check_name(row, HOST_COMPILER, "==", GCC)
            and int(row[param_map[HOST_COMPILER]][VERSION]) <= 6
        ):
            return False

    # GCC 8 and older does not support C++20
    if (
        is_in_row(row, CXX_STANDARD)
        and int(row[param_map[CXX_STANDARD]][VERSION]) >= 20
        and row_check_name(row, HOST_COMPILER, "==", GCC)
        and int(row[param_map[HOST_COMPILER]][VERSION]) <= 8
    ):
        return False

    if row_check_name(row, DEVICE_COMPILER, "==", NVCC) and is_in_row(
        row, CXX_STANDARD
    ):
        parsed_nvcc_version = pk_version.parse(row[param_map[DEVICE_COMPILER]][VERSION])

        # NVCC versions older than 11.0 does not support C++ 17
        if (
            parsed_nvcc_version < pk_version.parse("11.0")
            and int(row[param_map[CXX_STANDARD]][VERSION]) > 14
        ):
            return False

        # no NVCC version supports C++20
        if int(row[param_map[CXX_STANDARD]][VERSION]) > 17:
            return False

    # clang 11 and 12 are not available in the Ubuntu 18.04 ppa
    if (
        row_check_version(row, UBUNTU, "==", "18.04")
        and (
            row_check_name(
                row,
                HOST_COMPILER,
                "==",
                CLANG,
            )
            or row_check_name(
                row,
                HOST_COMPILER,
                "==",
                CLANG_CUDA,
            )
        )
        and (
            (
                row_check_version(row, HOST_COMPILER, "==", "11")
                or row_check_version(row, HOST_COMPILER, "==", "12")
            )
        )
    ):
        return False

    # ubuntu 18.04 containers are not available for CUDA 11.0 and later
    if (
        row_check_version(row, UBUNTU, "==", "18.04")
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF)
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">=", "11.0")
    ):
        return False

    # ubuntu 20.04 containers are not available for CUDA 10.2 and before
    if (
        row_check_version(row, UBUNTU, "==", "20.04")
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF)
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "<", "11.0")
    ):
        return False

    # all rocm images are Ubuntu 20.04 based
    if (
        row_check_version(row, UBUNTU, "!=", "20.04")
        and row_check_name(row, DEVICE_COMPILER, "==", HIPCC)
        and row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF)
    ):
        return False

    return True
