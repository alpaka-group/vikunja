from alpaka_job_coverage.util import is_in_row
from alpaka_job_coverage.util import row_check_name, row_check_version

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from vikunja_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


from packaging import version as pk_version


def vikunja_post_filter(row: List) -> bool:
    # TODO: FIXME, I'm a workaround, because vikunja does not work with the current development branch
    # and the CUDA backend
    if (
        is_in_row(row, ALPAKA)
        and is_in_row(row, DEVICE_COMPILER)
        and row[param_map[ALPAKA]][VERSION] == "develop"
        and row[param_map[DEVICE_COMPILER]][NAME] == NVCC
    ):
        return False

    # TODO: FIXME disable Clang hast CUDA host compiler
    if row_check_name(row, DEVICE_COMPILER, "==", NVCC) and row_check_name(
        row, HOST_COMPILER, "==", CLANG
    ):
        return False

    # GCC 9 and older and Clang 9 and older does not support official C++20 (max. -std=c++2a)
    if row_check_version(row, CXX_STANDARD, "==", "20"):
        for compiler in [HOST_COMPILER, DEVICE_COMPILER]:
            if row_check_version(row, compiler, "<", "10") and (
                row_check_name(row, compiler, "==", GCC)
                or row_check_name(row, compiler, "==", CLANG)
                or row_check_name(row, compiler, "==", CLANG_CUDA)
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

    return True
