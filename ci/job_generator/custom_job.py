"""Add custom jobs. For example loaded from a yaml file."""

import os, yaml
from typing import List, Dict
from typeguard import typechecked


@typechecked
def add_custom_jobs(job_matrix_yaml: List[Dict[str, Dict]], container_version: float):
    """Read custom jobs from yaml files and add it to the job_matrix_yaml.

    Args:
        job_matrix_yaml (List[Dict[str, Dict]]): The job matrix, containing the yaml code
        for each job.
        container_version (float): Used container version.

    Raises:
        RuntimeError: Throw error, if yaml file of custom jobs does not exits.
    """
    script_path = os.path.abspath(__file__)
    integration_test_path = os.path.abspath(
        os.path.join(
            os.path.dirname(script_path),
            "../custom_jobs/integration_test.yml",
        )
    )

    if not os.path.exists(integration_test_path):
        raise RuntimeError(f"{integration_test_path} does not exist")

    with open(integration_test_path, "r", encoding="utf8") as file:
        integration_jobs = yaml.load(file, yaml.loader.SafeLoader)

    for job_name, job_body in integration_jobs.items():
        job_body["image"] = job_body["image"] + ":" + str(container_version)
        job_matrix_yaml.insert(0, {job_name: job_body})
