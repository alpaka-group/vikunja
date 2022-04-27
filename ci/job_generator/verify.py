"""Verification of the results.
"""

from typing import List, Dict, Tuple
from typeguard import typechecked

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from vikunja_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import strict_equal
import versions


@typechecked
def verify(combinations: List[Dict[str, Tuple[str, str]]]) -> bool:
    """Check if job matrix fullfill certain requirements.

    At the moment, it checks only a software name or name version combination appears at least one
    time in a job.

    Args:
        combinations (List[Dict[str, Tuple[str, str]]]): The job matrix.

    Returns:
        bool: True if all checks passes, otherwise False.
    """

    expected_values: Dict[str, Dict[str, List[str]]] = {}

    expected_values[HOST_COMPILER] = {
        GCC: [],
        CLANG: [],
        CLANG_CUDA: [],
        HIPCC: [],
    }
    # I believe, nobody understand the syntax ;-p
    # it copies the the dict host_compiler and append the entry nvcc
    expected_values[DEVICE_COMPILER] = {
        **expected_values[HOST_COMPILER],
        **{NVCC: []},
    }

    expected_values[ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE] = {
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: ["on"],
    }
    expected_values[ALPAKA_ACC_GPU_CUDA_ENABLE] = {
        ALPAKA_ACC_GPU_CUDA_ENABLE: versions.sw_versions[NVCC],
    }
    expected_values[ALPAKA_ACC_GPU_HIP_ENABLE] = {
        ALPAKA_ACC_GPU_HIP_ENABLE: versions.sw_versions[HIPCC],
    }

    for sw_name in [CMAKE, BOOST, ALPAKA, UBUNTU, CXX_STANDARD]:
        expected_values[sw_name] = {
            sw_name: versions.sw_versions[sw_name],
        }

    # check if all values in expected_values exists at least on time in the job matrix
    for field in expected_values:
        for expected_name, expected_versions in expected_values[field].items():
            # if the version list is empty, check only the name
            if len(expected_versions) == 0:
                found = False
                for comb in combinations:
                    if field in comb and strict_equal(comb[field][NAME], expected_name):
                        found = True
                        break
                if not found:
                    print(f"\033[31m{expected_name} missing in {field}\033[m")
                    return False
            else:
                for expected_version in expected_versions:
                    found = False
                    for comb in combinations:
                        if (
                            field in comb
                            and strict_equal(comb[field][NAME], expected_name)
                            and strict_equal(comb[field][VERSION], expected_version)
                        ):
                            found = True
                            break
                    if not found:
                        print(
                            f"\033[31m{expected_name}@{expected_version} missing in {field}\033[m"
                        )
                        return False

    print("\033[32mverification was fine\033[m")

    return True
