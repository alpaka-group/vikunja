.. highlight:: none

CMake Arguments
===============

The value in the brackets after the argument type indicates the default value.

Common
++++++

**VIKUNJA_BUILD_EXAMPLES** (OFF)
    .. code-block::

        Build the examples.

**VIKUNJA_ENABLE_EXTRA_WARNINGS** (OFF)
    .. code-block::

        Set some additional compiler warning flags.

Testing
+++++++
.. _cmake-test:

**BUILD_TESTING** (OFF)
    .. code-block::

        Build the testing tree.

**VIKUNJA_SYSTEM_CATCH2** (OFF)
    .. code-block::

        Only works if BUILD_TESTING is ON.
        Use your local installation of Catch2.
        Otherwise, it will be automatically downloaded and installed in the local build folder.

**VIKUNJA_ENABLE_CXX_TEST** (OFF)
    .. code-block::

        Only works if BUILD_TESTING is ON.
        Special test that checks if ALPAKA_CXX_STANDARD works correctly.
        The implementation is very compiler specific, so it is possible that the test is not
        supported by your used C++ compiler.

**VIKUNJA_ENABLE_BENCHMARKS** (OFF)
    .. code-block::

        Only works if BUILD_TESTING is ON.
        Enable the benchmarks. The benchmarks are built automatically and can be executed via ctest.

alpaka
++++++

The following CMake variables are provided by alpaka. This section contains only the variables most important for vikunja. To see all variables refer to the `alpaka documentation <https://alpaka.readthedocs.io/en/latest/advanced/cmake.html>`_.

**ALPAKA_CXX_STANDARD** (17)
    .. code-block::

       Set the C++ standard version.

**ALPAKA_ACC_<ACC>_ENABLE** (OFF)
    .. code-block::

        Enable one or more accelerator backends. The following accelerators are available:
        - ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE
        - ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE
        - ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE
        - ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE
        - ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE
        - ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE
        - ALPAKA_ACC_ANY_BT_OMP5_ENABLE
        - ALPAKA_ACC_GPU_CUDA_ENABLE
        - ALPAKA_ACC_GPU_HIP_ENABLE

        Important: Not all alpaka accelerator backends are tested together with vikunja,
        see CI tests.

**ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA** (ON)
    .. code-block::

        Enable lambda support in Alpaka 0.6.x and below for the CUDA accelerator.

**ALPAKA_CUDA_EXPT_EXTENDED_LAMBDA** (ON)
    .. code-block::

        Enable lambda support in Alpaka 0.7.x and later for the CUDA accelerator.
