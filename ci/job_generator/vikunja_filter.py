from alpaka_job_coverage.util import is_in_row

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from vikunja_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


from packaging import version as pk_version


def vikunja_post_filter(row: List) -> bool:
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
