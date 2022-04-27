"""Filter rules basing on backend names and versions.
"""

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import (
    row_check_name,
    row_check_backend_version,
)


def compiler_backend_filter(row: List) -> bool:
    """Filter rules basing on backend names and versions.

    Args:
        row (List): Combination to verify. The row can contain
        up to all combination fields and at least two items.

    Returns:
        bool: True, if combination is valid, otherwise False.
    """
    ###########################
    ## gcc device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", GCC):
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF):
            return False

        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF):
            return False

    ###########################
    ## clang device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG):
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF):
            return False

        # clang cannot compile with enabled HIP backend
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF):
            return False

    ###########################
    ## nvcc device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", NVCC):
        # the nvcc compiler needs the same version, like the backend
        if row_check_backend_version(
            row,
            ALPAKA_ACC_GPU_CUDA_ENABLE,
            "!=",
            row[param_map[DEVICE_COMPILER]][VERSION],
        ):
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF):
            return False

    ###########################
    ## clang-cuda device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA):
        # the CUDA backend needs to be enabled
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "==", OFF):
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF):
            return False

    ###########################
    ## hipcc device compiler
    ###########################

    # the HIP backend needs to be enabled and has the same version number
    if row_check_name(row, DEVICE_COMPILER, "==", HIPCC):
        if row_check_backend_version(
            row,
            ALPAKA_ACC_GPU_HIP_ENABLE,
            "!=",
            row[param_map[DEVICE_COMPILER]][VERSION],
        ):
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF):
            return False

    return True
