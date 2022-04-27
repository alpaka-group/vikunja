"""Generate GitLab-CI test jobs yaml for the vikunja CI."""
import argparse
import sys
from typing import List, Dict, Tuple
from collections import OrderedDict

import alpaka_job_coverage as ajc
from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

from vikunja_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from versions import (
    get_sw_tuple_list,
    get_compiler_versions,
    get_backend_matrix,
)
from vikunja_filter import vikunja_post_filter
from reorder_jobs import reorder_jobs
from generate_job_yaml import generate_job_yaml
from verify import verify


def get_args() -> argparse.Namespace:
    """Define and parse the commandline arguments.

    Returns:
        argparse.Namespace: The commandline arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate job matrix and create GitLab CI .yml."
    )

    parser.add_argument(
        "version", type=float, help="Version number of the used CI container."
    )
    parser.add_argument(
        "--print-combinations",
        action="store_true",
        help="Display combination matrix.",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify generated combination matrix"
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Combine flags: --print-combinations and --verify",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="./jobs.yml",
        help="Path of the generated jobs yaml.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # setup the parameters
    parameters: OrderedDict = OrderedDict()
    parameters[HOST_COMPILER] = get_compiler_versions()
    parameters[DEVICE_COMPILER] = get_compiler_versions()
    parameters[BACKENDS] = get_backend_matrix()
    parameters[CMAKE] = get_sw_tuple_list(CMAKE)
    parameters[BOOST] = get_sw_tuple_list(BOOST)
    parameters[ALPAKA] = get_sw_tuple_list(ALPAKA)
    parameters[UBUNTU] = get_sw_tuple_list(UBUNTU)
    parameters[CXX_STANDARD] = get_sw_tuple_list(CXX_STANDARD)

    job_matrix: List[Dict[str, Tuple[str, str]]] = ajc.create_job_list(
        parameters=parameters,
        post_filter=vikunja_post_filter,
        pair_size=2,
    )

    if args.print_combinations or args.all:
        print(f"number of combinations before reorder: {len(job_matrix)}")

    ajc.shuffle_job_matrix(job_matrix)
    reorder_jobs(job_matrix)

    if args.print_combinations or args.all:
        for compiler in job_matrix:
            print(compiler)

        print(f"number of combinations: {len(job_matrix)}")

    # TODO: remove max number of jobs
    wave_job_matrix = ajc.distribute_to_waves(job_matrix, 10)

    if args.verify or args.all:
        if not verify(job_matrix):
            sys.exit(1)

    generate_job_yaml(
        job_matrix=wave_job_matrix,
        path=args.output_path,
        container_version=args.version,
    )
