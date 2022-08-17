from alpaka_job_coverage.util import is_in_row
from alpaka_job_coverage.util import row_check_name, row_check_version

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from vikunja_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


from packaging import version as pk_version


def vikunja_post_filter(row: List) -> bool:
    # TODO: FIXME disable Clang hast CUDA host compiler
    if row_check_name(row, DEVICE_COMPILER, "==", NVCC) and row_check_name(
        row, HOST_COMPILER, "==", CLANG
    ):
        return False

    # the minimum boost version for alpaka 0.9.0 is 1.74.0
    if (
        is_in_row(row, ALPAKA)
        and is_in_row(row, BOOST)
        and (
            pk_version.parse(row[param_map[ALPAKA]][VERSION])
            > pk_version.parse("0.8.0")
            or row[param_map[ALPAKA]][VERSION] == "develop"
        )
        and pk_version.parse(row[param_map[BOOST]][VERSION])
        < pk_version.parse("1.74.0")
    ):
        return False

    # CUDA 11.3+ is only supported by alpaka 0.7.0 an newer
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_version(row, DEVICE_COMPILER, ">=", "11.3")
        and row_check_version(row, ALPAKA, "<", "0.7.0")
    ):
        return False

    # disable HIP for alpaka 0.8.0 and older, because of missing HIP 4.3+ support
    if row_check_name(row, DEVICE_COMPILER, "==", HIPCC) and row_check_version(
        row, ALPAKA, "<", "0.9.0"
    ):
        return False

    return True
