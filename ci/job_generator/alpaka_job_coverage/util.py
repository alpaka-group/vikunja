"""Different support functions.
"""

import operator
from typing import Any, List, Dict, Tuple
from typeguard import typechecked
from packaging import version as pk_version


from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

# maps strings to comparision operators
OPERATOR_MAP = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


# no typechecked, because function is performance critical
def strict_equal(first_value: Any, second_value: Any) -> bool:
    """Compare types and values of a and b. If the types are different,
    throw error. If the types are equal the result is a bool.

    Args:
        first_value (Any): first value to compare
        second_value (Any): second value to compare

    Raises:
        TypeError: Is thrown, if the types of a and b are different.

    Returns:
        bool: True if values are equal, otherwise False.
    """
    if not isinstance(first_value, type(second_value)):
        raise TypeError(
            f"a and b has not the same type: {type(first_value)} != {type(second_value)}"
        )

    return first_value == second_value


def is_in_row(row: List, name: str) -> bool:
    """Check if paramater is in the row.

    Args:
        row (List): Row with parameters.
        name (str): The searched parameter.

    Returns:
        bool: Return True, if parameter is in row.
    """
    return param_map[name] < len(row)


# no typechecked, because function is performance critical
def row_check_name(row: List, colum: str, opr: str, name: str) -> bool:
    """Check if colum is in row and if the name matches or not, depending
    of the operator.

    Args:
        row (List): Row to check.
        colum (str): Colum name in the row.
        opr (str): The operator can be "==" (equal) or "!=" (not equal).
        name (str): Name to compare.

    Raises:
        ValueError: Raise error, if operator does not have the value "==" or "!=".

    Returns:
        bool: Return False, if column is not in the row. If the column is in the row,
        return True if the name matches ("==") or not matches ("!=").
    """
    if not opr in ("==", "!="):
        raise ValueError("op (operator) needs to be == or !=")

    return is_in_row(row, colum) and OPERATOR_MAP[opr](
        row[param_map[colum]][NAME], name
    )


# no typechecked, because function is performance critical
def row_check_version(
    row: List,
    colum: str,
    opr: str,
    version: str,
) -> bool:
    """Check if colum is in row and if the version matches or not, depending
    of the operator.

    Args:
        row (List): Row to check.
        colum (str): Colum name in the row.
        opr (str): The operator can be "==", "!=", "<", "<=", ">" and ">=".
        version (str): Version to compare.

    Raises:
        ValueError: Raise error, if operator does not have the supported value.

    Returns:
        bool: Return False, if column is not in the row. If the column is in the row,
        return True if the version.
    """
    if not opr in OPERATOR_MAP:
        raise ValueError(f"operator needs to be: {', '.join(OPERATOR_MAP.keys())}")

    return is_in_row(row, colum) and OPERATOR_MAP[opr](
        pk_version.parse(row[param_map[colum]][VERSION]), pk_version.parse(version)
    )


# no typechecked, because function is performance critical
def row_check_backend_version(row: List, backend: str, opr: str, version: str) -> bool:
    """Check, if backend exists and if the backend version matches depending of the operator.

    Args:
        row (List): Row to check.
        backend (str): Name of the backend, which version should be compared.
        opr (str): The operator can be "==", "!=", "<", "<=", ">" and ">=".
        version (str): Version to compare.

    Raises:
        ValueError: Raise error, if operator does not have the supported value.

    Returns:
        bool: Return False, if backend does not exist. If the backend name is in the row, return
        True if the version matches.
    """
    if not opr in OPERATOR_MAP:
        raise ValueError(f"operator needs to be: {', '.join(OPERATOR_MAP.keys())}")

    if not is_in_row(row, BACKENDS):
        return False

    for row_backend in row[param_map[BACKENDS]]:
        if row_backend[NAME] == backend:
            return OPERATOR_MAP[opr](
                pk_version.parse(row_backend[VERSION]), pk_version.parse(version)
            )

    return False


@typechecked
def search_and_move_job(
    job_matrix: List[Dict[str, Tuple[str, str]]],
    searched_job: Dict[str, Tuple[str, str]],
    position: int = 0,
) -> bool:
    """Search job, which contains all items of searched_job and move it to list position.

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): The job matrix.
        searched_job (Dict[str, Tuple[str, str]]): Dict of searched items. If all items matches with
        an entry in job list, move entry to position
        position (int, optional): New position of matched entry in job_matrix. Defaults to 0.

    Raises:
        IndexError: Raise error, if searched_job dict is empty.

    Returns:
        bool: True if found and move entry, otherwise False. If False, job_matrix was not modified.
    """
    if len(searched_job) == 0:
        raise IndexError("searched_job must not be empty")

    for index, job_combination in enumerate(job_matrix):
        matched_attributes = 0
        for attribute_name, attribute_value in searched_job.items():
            if (
                attribute_name in job_combination
                and job_combination[attribute_name] == attribute_value
            ):
                matched_attributes += 1
            if matched_attributes == len(searched_job):
                job_matrix.insert(position, job_matrix.pop(index))
                return True
    return False
