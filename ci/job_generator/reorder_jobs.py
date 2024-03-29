"""Functions to modify order of the job list.
"""

from typing import List, Dict, Tuple

from typeguard import typechecked

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import search_and_move_job
from vikunja_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from versions import sw_versions

COUNTER = 0


def force_job(
    job_matrix: List[Dict[str, Tuple[str, str]]],
    searched_job: Dict[str, Tuple[str, str]],
) -> bool:
    """Search and move job to the first position. If the job can not be found, create a new job. The
    function has an internal counter for the position in the job_matrix. The jobs will be moved or
    insert in the same order, like the function is called.

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): The job Matrix
        searched_job (Dict[str, Tuple[str, str]]): Dict of search requirements.

    Returns:
        bool: return True if successful.
    """
    global COUNTER  # pylint: disable=global-statement

    new_job: Dict[str, Tuple[str, str]] = {
        HOST_COMPILER: ("", ""),
        DEVICE_COMPILER: ("", ""),
        CMAKE: (CMAKE, sw_versions[CMAKE][-1]),
        BOOST: (BOOST, sw_versions[BOOST][-1]),
        ALPAKA: (ALPAKA, sw_versions[ALPAKA][-1]),
        UBUNTU: (UBUNTU, "20.04"),
        CXX_STANDARD: (CXX_STANDARD, "17"),
    }

    if not search_and_move_job(
        job_matrix=job_matrix,
        searched_job=searched_job,
        position=COUNTER,
    ):
        for key, value in searched_job.items():
            new_job[key] = value
        job_matrix.insert(COUNTER, new_job)

    COUNTER = COUNTER + 1
    return True


@typechecked
def reorder_jobs(job_matrix: List[Dict[str, Tuple[str, str]]]):
    """Vikunja specific function, to move jobs, which matches certain properties to the first waves.

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): The job_matrix.
    """

    force_job(
        job_matrix=job_matrix,
        searched_job={
            HOST_COMPILER: (GCC, "9"),
            DEVICE_COMPILER: (GCC, "9"),
            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                ON,
            ),
            # latest release
            ALPAKA: (ALPAKA, sw_versions[ALPAKA][-2]),
        },
    )

    force_job(
        job_matrix=job_matrix,
        searched_job={
            HOST_COMPILER: (GCC, "9"),
            DEVICE_COMPILER: (GCC, "9"),
            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                ON,
            ),
            # develop
            ALPAKA: (ALPAKA, sw_versions[ALPAKA][-1]),
        },
    )

    force_job(
        job_matrix=job_matrix,
        searched_job={
            # latest clang version
            HOST_COMPILER: (CLANG, sw_versions[CLANG][-1]),
            DEVICE_COMPILER: (CLANG, sw_versions[CLANG][-1]),
            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                ON,
            ),
            ALPAKA: (ALPAKA, sw_versions[ALPAKA][-2]),
        },
    )

    force_job(
        job_matrix=job_matrix,
        searched_job={
            HOST_COMPILER: (CLANG, sw_versions[CLANG][-1]),
            DEVICE_COMPILER: (CLANG, sw_versions[CLANG][-1]),
            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                ON,
            ),
            ALPAKA: (ALPAKA, sw_versions[ALPAKA][-1]),
        },
    )

    force_job(
        job_matrix=job_matrix,
        searched_job={
            HOST_COMPILER: (GCC, "9"),
            # latest NVCC version
            DEVICE_COMPILER: (NVCC, sw_versions[NVCC][-1]),
            ALPAKA_ACC_GPU_CUDA_ENABLE: (
                ALPAKA_ACC_GPU_CUDA_ENABLE,
                sw_versions[NVCC][-1],
            ),
            ALPAKA: (ALPAKA, sw_versions[ALPAKA][-2]),
        },
    )

    force_job(
        job_matrix=job_matrix,
        searched_job={
            HOST_COMPILER: (GCC, "9"),
            DEVICE_COMPILER: (NVCC, sw_versions[NVCC][-1]),
            ALPAKA_ACC_GPU_CUDA_ENABLE: (
                ALPAKA_ACC_GPU_CUDA_ENABLE,
                sw_versions[NVCC][-1],
            ),
            ALPAKA: (ALPAKA, sw_versions[ALPAKA][-1]),
        },
    )

    force_job(
        job_matrix=job_matrix,
        searched_job={
            HOST_COMPILER: (HIPCC, sw_versions[HIPCC][-1]),
            DEVICE_COMPILER: (HIPCC, sw_versions[HIPCC][-1]),
            ALPAKA_ACC_GPU_HIP_ENABLE: (
                ALPAKA_ACC_GPU_HIP_ENABLE,
                sw_versions[HIPCC][-1],
            ),
            ALPAKA: (ALPAKA, sw_versions[ALPAKA][-2]),
        },
    )

    force_job(
        job_matrix=job_matrix,
        searched_job={
            HOST_COMPILER: (HIPCC, sw_versions[HIPCC][-1]),
            DEVICE_COMPILER: (HIPCC, sw_versions[HIPCC][-1]),
            ALPAKA_ACC_GPU_HIP_ENABLE: (
                ALPAKA_ACC_GPU_HIP_ENABLE,
                sw_versions[HIPCC][-1],
            ),
            ALPAKA: (ALPAKA, sw_versions[ALPAKA][-1]),
        },
    )
