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
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF_VER
    ):
        # Cast cuda version shape. E.g. from 11.0 to 110
        container_url += "-cuda" + str(
            int(float(job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION]) * 10)
        )
        if job[HOST_COMPILER][NAME] == GCC:
            container_url += "-gcc"

    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF_VER
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

    if job[DEVICE_COMPILER][NAME] != CLANG_CUDA:
        # Set variables for host and device compiler name and version to check if the compilers are
        # available or needs to be installed
        variables[f"VIKUNJA_CI_{get_env_var_name(job[HOST_COMPILER][NAME])}_VER"] = job[
            HOST_COMPILER
        ][VERSION]
        variables[
            f"VIKUNJA_CI_{get_env_var_name(job[DEVICE_COMPILER][NAME])}_VER"
        ] = job[DEVICE_COMPILER][VERSION]
    else:
        variables[f"VIKUNJA_CI_{get_env_var_name(CLANG)}_VER"] = job[HOST_COMPILER][
            VERSION
        ]
        variables[f"VIKUNJA_CI_{get_env_var_name(CLANG)}_VER"] = job[DEVICE_COMPILER][
            VERSION
        ]

    # required to check, if the correct CUDA SDK is available
    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF_VER
    ):
        variables["VIKUNJA_CI_CUDA_VER"] = job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION]

    # required to check, if the correct ROCm SDK is available
    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF_VER
    ):
        variables["VIKUNJA_CI_ROCM_VER"] = job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION]

    # create CMake string to enable the alpaka backends
    backend_str = ""
    for backend in BACKENDS_LIST:
        if backend in job:
            # since alpaka 0.9.0, the backend names starts with a lowercase `alpaka`
            if pk_version.parse(job[ALPAKA][VERSION]) > pk_version.parse("0.8.0"):
                backend_name = job[backend][NAME]
            else:
                backend_name = job[backend][NAME].upper()

            if job[backend][VERSION] == OFF_VER:
                backend_str += f"-D{backend_name}=OFF "
            else:
                backend_str += f"-D{backend_name}=ON "

    if backend_str != "":
        variables["VIKUNJA_CI_ALPAKA_BACKENDS"] = backend_str

    # simply copy name and version of the required software
    for parameter_name in [CMAKE, BOOST, ALPAKA]:
        if parameter_name == ALPAKA and job[parameter_name][VERSION].endswith("-dev"):
            # the current alpaka development version is not tagged, therefore use the development branch
            variables[f"VIKUNJA_CI_{get_env_var_name(parameter_name)}_VER"] = "develop"
        else:
            variables[f"VIKUNJA_CI_{get_env_var_name(parameter_name)}_VER"] = job[
                parameter_name
            ][VERSION]

    variables["VIKUNJA_CI_CXX_STANDARD"] = job[CXX_STANDARD][VERSION]

    # static variables
    # is required for a recursive clone
    variables["GIT_SUBMODULE_STRATEGY"] = "normal"

    cmake_extra_arg = []

    if job[DEVICE_COMPILER][NAME] == CLANG_CUDA:
        cmake_extra_arg.append(
            f"-DCMAKE_CUDA_COMPILER=clang++-{job[DEVICE_COMPILER][VERSION]}"
        )

    if job[DEVICE_COMPILER][NAME] == NVCC:
        cmake_extra_arg.append(
            f'-DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_HOST_COMPILER={variables["VIKUNJA_CI_CXX"]}'
        )

    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF_VER
    ):
        # architecture of the Vega 64
        cmake_extra_arg.append("-DGPU_TARGETS=${CI_GPU_ARCH}")

    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF_VER
    ):
        # CI contains a Quadro P5000 (sm_61)
        cmake_extra_arg.append("-DCMAKE_CUDA_ARCHITECTURES=61")

    # enable cxx version test for only for specific compiler
    # * for C++ 17 every supported compiler
    # * for C++ 20
    #   * GCC 11 and newer
    #   * Clang 10 and newer
    enable_cxx_test = False
    if CXX_STANDARD in job and job[CXX_STANDARD][VERSION] == "17":
        enable_cxx_test = True
    elif CXX_STANDARD in job and job[CXX_STANDARD][VERSION] == "20":
        for compiler_name in [HOST_COMPILER, DEVICE_COMPILER]:
            if compiler_name in job and (
                (
                    (job[compiler_name][NAME] == GCC)
                    and pk_version.parse(job[compiler_name][VERSION])
                    >= pk_version.parse("11.0")
                )
                or (
                    job[compiler_name][NAME] == CLANG
                    and pk_version.parse(job[compiler_name][VERSION])
                    >= pk_version.parse("10.0")
                )
            ):
                enable_cxx_test = True

    if enable_cxx_test:
        variables["VIKUNJA_CI_CXX_TEST"] = "ON"
    else:
        variables["VIKUNJA_CI_CXX_TEST"] = "OFF"

    variables["VIKUNJA_CI_CATCH_VER"] = "v2.13.2"

    variables["VIKUNJA_CI_EXTRA_ARGS"] = " ".join(cmake_extra_arg)
    variables["VIKUNJA_CI_CONST_ARGS"] = " ".join(
        [
            "-DBUILD_TESTING=ON",
            "-DVIKUNJA_SYSTEM_CATCH2=ON",
            "-DVIKUNJA_BUILD_EXAMPLES=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )

    return variables


@typechecked
def job_tags(job: Dict[str, Tuple[str, str]]) -> List[str]:
    """Add tags to select the correct runner, e.g. CPU only or Nvidia GPU.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.

    Returns:
        List[str]: List of tags.
    """
    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF_VER
    ):
        return ["x86_64", "cuda"]
    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF_VER
    ):
        return ["x86_64", "rocm"]
    return ["x86_64", "cpuonly"]


@typechecked
def create_job(
    job: Dict[str, Tuple[str, str]], container_version: float
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
    # if Clang-CUDA is the device compiler, add also the CUDA SDK version to the name
    if job[DEVICE_COMPILER][NAME] == CLANG_CUDA:
        job_name = job_name + "-CUDA" + job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION]
    job_name += job_prefix_coding(job)

    job_yaml: Dict = {}

    job_yaml["image"] = job_image(job, container_version)
    job_yaml["variables"] = job_variables(job)
    job_yaml["script"] = [
        "source ./ci/gitlab_scripts/setup.sh",
        "source ./ci/gitlab_scripts/print_env.sh",
        "source ./ci/gitlab_scripts/test.sh",
    ]
    job_yaml["tags"] = job_tags(job)

    return {job_name: job_yaml}


@typechecked
def generate_job_yaml_list(
    job_matrix: List[Dict[str, Tuple[str, str]]],
    container_version: float,
) -> List[Dict[str, Dict]]:
    """Generate the job yaml for each job in the job matrix.

    Args:
        job_matrix (List[List[Dict[str, Tuple[str, str]]]]): Job Matrix
        container_version (float): Container version tag.

    Returns:
        List[Dict[str, Dict]]: List of GitLab-CI jobs. The key of a dict entry
        is the job name and the value is the body.
    """
    job_matrix_yaml: Dict[str, Dict] = []
    for job in job_matrix:
        job_matrix_yaml.append(create_job(job, container_version))

    return job_matrix_yaml


@typechecked
def write_job_yaml(
    job_matrix: List[List[Dict[str, Dict]]],
    path: str,
):
    """Write GitLab-CI jobs to file.

    Args:
        job_matrix (List[List[Dict[str, Dict]]]): List of GitLab-CI jobs. The
        key of a dict entry is the job name and the value is the body.
        path (str): Path of the GitLab-CI yaml file.
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
                # the first key is the name
                job[list(job.keys())[0]]["stage"] = "stage" + str(stage_number)

                yaml.dump(job, output_file)
                output_file.write("\n")
