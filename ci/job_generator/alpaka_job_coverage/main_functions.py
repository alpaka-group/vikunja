"""API for creating sparse job list from parameters list.
"""

import random
from typing import Dict, List, Tuple, Callable

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.filter_compiler_name import general_compiler_filter
from alpaka_job_coverage.filter_compiler_version import compiler_version_filter
from alpaka_job_coverage.filter_backend_version import compiler_backend_filter
from alpaka_job_coverage.filter_software_dependency import (
    software_dependency_filter,
)

from typeguard import typechecked

from allpairspy import AllPairs


@typechecked
def create_job_list(  # pylint: disable=too-many-arguments
    parameters: Dict[str, List],
    pair_size: int = 2,
    pre_filter: Callable[[List], bool] = lambda row: True,
    after_gerneral_compiler_filter: Callable[[List], bool] = lambda row: True,
    after_compiler_version_filter: Callable[[List], bool] = lambda row: True,
    post_filter: Callable[[List], bool] = lambda row: True,
    after_compiler_backend_filter: Callable[[List], bool] = lambda row: True,
) -> List[Dict[str, Tuple[str, str]]]:
    """Create sparse job list from parameters list, respecting the different filter lists.

    Order of the filter functions:
        pre_filter()
        general_compiler_filter()
        after_gerneral_compiler_filter()
        compiler_version_filter()
        after_compiler_version_filter()
        compiler_backend_filter()
        post_filter()

    Args:
        parameters (Dict[str, List]): Dict of parameters with List of values. The keys
        "host_compiler", "device_compiler" and "backends" are required. Each element in the list of
        a dict items needs to be a tuple[str, str]. The first value is the name and second value is
        the version. Only "backends" is an exception. The type of the value of a "backends" item is
        a List[Tuple[str, str]]. Each tuple contains a backend name and version.

        pair_size (int, optional): Each combination of the values of n different parameter fields
        needs to part of at least one job. If there are more than n fields in parameter, it uses
        each possible n combination of the field name. For example, if n is 3 and there are 4
        fields, it will use the following combinations (the numbers represents the field names):
        (0,1,2), (0,1,3), (0,2,3), ... (2,3,4). Than it take the values of each filed of the
        combination and create the full combination matrix. Each of
        this value combinations needs be part of a job. Defaults is 2.

        pre_filter (Callable[[List], bool], optional): This filter is executed first.
        Defaults is a lambda which returns true.

        after_gerneral_compiler_filter (Callable[[List], bool], optional): This
        filter is executed after the general_compiler_filter. Defaults is a lambda which returns
        true.

        after_compiler_version_filter (Callable[[List], bool], optional): This filter
        is executed after the general_compiler_filter. Defaults is a lambda which returns true.

        post_filter (Callable[[List], bool], optional): This filter is executed last.
        Defaults is a lambda which returns true.

    Returns:
        List[Dict[str, List[Tuple[str, str]]]]: Return sparse job matrix.
    """

    # Fill up the param_map.
    # For documentation see the variable documentation in globals.py.
    for index, key in enumerate(parameters):
        param_map[key] = index

    # build filter chain
    filter_function = (
        lambda row: pre_filter(row)
        and general_compiler_filter(row)
        and after_gerneral_compiler_filter(row)
        and compiler_version_filter(row)
        and after_compiler_version_filter(row)
        and compiler_backend_filter(row)
        and after_compiler_backend_filter(row)
        and software_dependency_filter(row)
        and post_filter(row)
    )

    # generate sparse test matrix
    cover_matrix = AllPairs(
        parameters=parameters, n=pair_size, filter_func=filter_function
    )

    # transform the sparse test matrix in a flat data struct
    normalized_cover_matrix: List[Dict[str, Tuple[str, str]]] = []
    for row in cover_matrix:
        new_row = {}
        for index, parameter_name in enumerate(parameters.keys()):
            if parameter_name != BACKENDS:
                new_row[parameter_name] = row[index]
            else:
                # Remove the column backend with its list of alpaka backends and insert each alpaka
                # backend in a own column.
                for backend in row[index]:
                    new_row[backend[NAME]] = backend
        normalized_cover_matrix.append(new_row)

    return normalized_cover_matrix


@typechecked
def shuffle_job_matrix(job_matrix: List[Dict[str, Tuple[str, str]]], seed: int = 42):
    """Shuffle ordering of the job_matrix.

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): The job_matrix.
        seed (int, optional): Seed of the shuffle function. Defaults to 42.
    """
    random.Random(seed).shuffle(job_matrix)


@typechecked
def distribute_to_waves(
    job_matrix: List[Dict[str, Tuple[str, str]]], wave_size: int, max_jobs: int = 0
) -> List[List[Dict[str, Tuple[str, str]]]]:
    """Distribute the jobs of job_matrix (1D) to a list of waves (2D). Each wave has the size
    wave_size. Total number in list waves is smaller or equal to max_jobs.

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): The job_matrix.
        wave_size (int): Size of each wave.
        max_jobs (int, optional): Maximum number of jobs in the returned matrix. Defaults to 0.

    Returns:
        List[List[Dict[str, Tuple[str, str]]]]: 2D Matrix with wave_size jobs each outer list.
    """
    # if max_jobs is 0, set to len(job_matrix)
    max_jobs = max_jobs if max_jobs > 0 else len(job_matrix)

    # if max_jobs is greater than len(job_matrix), crop it to len(job_matrix)
    max_jobs = max_jobs if max_jobs < len(job_matrix) else len(job_matrix)

    wave_matrix: List[List[Dict[str, Tuple[str, str]]]] = []

    for i in range(0, max_jobs, wave_size):
        wave_matrix.append(job_matrix[i : i + wave_size])

    # if wave_size is not a multiple of max_jobs and max_jobs is smaller than len(job_matrix),
    # the last wave contains to much jobs
    # this statement removes the jobs that are too much
    if max_jobs % wave_size != 0 and len(wave_matrix[-1]) > max_jobs % wave_size:
        wave_matrix[-1] = wave_matrix[-1][: max_jobs % wave_size]

    return wave_matrix
