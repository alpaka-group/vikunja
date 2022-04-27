"""Create GitLab-CI job description written in yaml from the job matrix."""

from typing import List, Dict, Tuple
from typeguard import typechecked
import yaml
from packaging import version as pk_version


from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from vikunja_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


@typechecked
def get_env_var_name(variable_name: str) -> str:
    """Transform string to a shape, which is allowed as environment variable.

    Args:
        variable_name (str): Variable name.

    Returns:
        str: Transformed variable name.
    """
    return variable_name.upper().replace("-", "_")


@typechecked
def job_prefix_coding(job: Dict[str, Tuple[str, str]]) -> str:
    """Generate prefix for job name, depending of the available software versions.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.

    Returns:
        str: Job name Prefix.
    """
    version_str = ""
    for sw in [CMAKE, BOOST, ALPAKA, UBUNTU, CXX_STANDARD]:
        if sw in job:
            if job[sw][NAME] == CXX_STANDARD:
                version_str += "_cxx" + job[sw][VERSION]
            else:
                version_str += "_" + job[sw][NAME] + job[sw][VERSION]

    return version_str


@typechecked
def job_image(job: Dict[str, Tuple[str, str]], container_version: float) -> str:
    """Generate the image url deppending on the host and device compiler and the selected backend
    of the job.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.
        container_version (float): Container version tag.

    Returns:
        str: Full container url, which can be used with docker pull.
    """
    container_url = "registry.hzdr.de/crp/alpaka-group-container/"
    container_url += "alpaka-ci-ubuntu" + job[UBUNTU][VERSION]

    # If only the GCC is used, use special gcc version of the container.
    if job[HOST_COMPILER][NAME] == GCC and job[DEVICE_COMPILER][NAME] == GCC:
        container_url += "-gcc"

    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF
    ):
        # Cast cuda version shape. E.g. from 11.0 to 110
        container_url += "-cuda" + str(
            int(float(job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION]) * 10)
        )
        if job[HOST_COMPILER][NAME] == GCC:
            container_url += "-gcc"

    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF
    ):
        container_url += "-rocm" + job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION]

    # append container tag
    container_url += ":" + str(container_version)
    return container_url


@typechecked
def job_variables(job: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """Add variables to the job depending of the job dict.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict

    Returns:
        Dict[str, str]: Dict of {variable name : variable value}.
    """
    variables: Dict[str, str] = {}

    # add variables with correct names, that can be called
    # e.g. -DCMAKE_CXX_COMPILER=${VIKUNJA_CI_CXX}
    for compiler in [HOST_COMPILER, DEVICE_COMPILER]:
        compiler_variable_name = (
            "VIKUNJA_CI_CXX" if compiler == HOST_COMPILER else "VIKUNJA_CI_DEVICE_CXX"
        )
        if job[compiler][NAME] == GCC:
            variables[compiler_variable_name] = "g++-" + job[compiler][VERSION]

        if job[compiler][NAME] in (CLANG, CLANG_CUDA):
            variables[compiler_variable_name] = "clang++-" + job[compiler][VERSION]

        if job[compiler][NAME] in (NVCC, HIPCC):
            variables[compiler_variable_name] = job[compiler][NAME]

    # Set variables for host and device compiler name and version to check if the compilers are
    # available or needs to be installed
    variables[f"VIKUNJA_CI_{get_env_var_name(job[HOST_COMPILER][NAME])}_VER"] = job[
        HOST_COMPILER
    ][VERSION]
    variables[f"VIKUNJA_CI_{get_env_var_name(job[DEVICE_COMPILER][NAME])}_VER"] = job[
        DEVICE_COMPILER
    ][VERSION]

    # required to check, if the correct CUDA SDK is available
    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF
    ):
        variables["VIKUNJA_CI_CUDA_VER"] = job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION]

    # required to check, if the correct ROCm SDK is available
    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF
    ):
        variables["VIKUNJA_CI_ROCM_VER"] = job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION]

    # create CMake string to enable the alpaka backends
    backend_str = ""
    for backend in BACKENDS_LIST:
        if backend in job:
            # since alpaka 0.9.0, the backend names starts with a lowercase `alpaka`
            if job[ALPAKA][VERSION] == "develop" or pk_version.parse(
                job[ALPAKA][VERSION]
            ) > pk_version.parse("0.8.0"):
                backend_name = job[backend][NAME]
            else:
                backend_name = job[backend][NAME].upper()

            if job[backend][VERSION] == OFF:
                backend_str += f"-D{backend_name}=OFF "
            else:
                backend_str += f"-D{backend_name}=ON "

    if backend_str != "":
        variables["VIKUNJA_CI_ALPAKA_BACKENDS"] = backend_str

    # simply copy name and version of the required software
    for parameter_name in [CMAKE, BOOST, ALPAKA]:
        variables[f"VIKUNJA_CI_{get_env_var_name(parameter_name)}_VER"] = job[
            parameter_name
        ][VERSION]

    variables["VIKUNJA_CI_CXX_STANDARD"] = job[CXX_STANDARD][VERSION]

    # static variables
    # is required for a recursive clone
    variables["GIT_SUBMODULE_STRATEGY"] = "normal"

    cmake_extra_arg = ""

    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF
    ):
        # architecture of the Vega 64
        cmake_extra_arg += "-DALPAKA_HIP_ARCH=900 "

    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF
    ):
        # CI contains a Quadro P5000 (sm_61)
        cmake_extra_arg += "-DCMAKE_CUDA_ARCHITECTURES=61 "

    variables["VIKUNJA_CI_EXTRA_ARGS"] = cmake_extra_arg

    return variables


@typechecked
def job_tags(job: Dict[str, Tuple[str, str]]) -> List[str]:
    """Add tags to select the correct runner, e.g. CPU only or Nvidia GPU.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.

    Returns:
        List[str]: List of tags.
    """
    # TODO: change back to correct tags
    return ["x86_64", "cpuonly"]

    if job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF:
        return ["x86_64", "cuda"]
    if job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF:
        return ["x86_64", "rocm"]
    return ["x86_64", "cpuonly"]


@typechecked
def create_job(
    job: Dict[str, Tuple[str, str]], stage_number: int, container_version: float
) -> Dict[str, Dict]:
    """Create complete GitLab-CI yaml for a single job

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.
        stage_number (int): Number of the stage. Required for the stage attribute.
        container_version (float): Container version tag.

    Returns:
        Dict[str, Dict]: Job yaml.
    """

    # the job name starts with the device compiler
    job_name = job[DEVICE_COMPILER][NAME].upper() + job[DEVICE_COMPILER][VERSION]
    # if the nvcc is the device compiler, add also the host compiler to the name
    if job[DEVICE_COMPILER][NAME] == NVCC:
        job_name = (
            job_name
            + "-"
            + job[HOST_COMPILER][NAME].upper()
            + job[HOST_COMPILER][VERSION]
        )
    job_name += job_prefix_coding(job)

    job_yaml: Dict = {}

    job_yaml["image"] = job_image(job, container_version)
    job_yaml["stage"] = "stage" + str(stage_number)
    job_yaml["variables"] = job_variables(job)
    job_yaml["script"] = [
        "./ci/gitlab_scripts/setup.sh",
        "./ci/gitlab_scripts/print_env.sh",
        "./ci/gitlab_scripts/test.sh",
    ]
    job_yaml["tags"] = job_tags(job)

    return {job_name: job_yaml}


@typechecked
def generate_job_yaml(
    job_matrix: List[List[Dict[str, Tuple[str, str]]]],
    path: str,
    container_version: float,
):
    """Generate the job yaml for each job in the job matrix and write it ot a file.

    Args:
        job_matrix (List[List[Dict[str, Tuple[str, str]]]]): Job Matrix
        path (str): Path of the GitLab-CI yaml file.
        container_version (float): Container version tag.
    """
    with open(path, "w", encoding="utf-8") as output_file:
        # setup all stages
        stages: Dict[str, List[str]] = {"stages": []}
        for stage_number in range(len(job_matrix)):
            stages["stages"].append(f"stage{stage_number}")
        yaml.dump(stages, output_file)
        output_file.write("\n")

        # Writes each job separately to the file.
        # If all jobs would be collected first in dict, the order would be not guarantied.
        for stage_number, wave in enumerate(job_matrix):
            # Improve the readability of the generated job yaml
            output_file.write(f"# <<<<<<<<<<<<< stage {stage_number} >>>>>>>>>>>>>\n\n")
            for job in wave:
                yaml.dump(create_job(job, stage_number, container_version), output_file)
                output_file.write("\n")
