.. highlight:: none

CMake Arguments
===============

Common
++++++

VIKUNJA_BUILD_EXAMPLES
    .. code-block::

        Build the examples.

VIKUNJA_ENABLE_EXTRA_WARNINGS
    .. code-block::

        Set some additional compiler warning flags.

Testing
+++++++

BUILD_TESTING
    .. code-block::

        Build the testing tree.

VIKUNJA_SYSTEM_CATCH2
    .. code-block::

        Only works if BUILD_TESTING is ON.
        Use your local installation of Catch2 v3.
        Otherwise, it will be automatically downloaded and installed in the local build folder.

VIKUNJA_ENABLE_CXX_TEST
    .. code-block::

        Only works if BUILD_TESTING is ON.
        Special test that checks if ALPAKA_CXX_STANDARD works correctly.
        The implementation is very compiler specific, so it is possible that the test is not supported by your used C++ compiler.

Alpaka
++++++

The following cmake variables are provided by alpaka. This section contains only the most important variables. To see all variables, see `here <https://alpaka.readthedocs.io/en/latest/advanced/cmake.html>`_.

ALPAKA_CXX_STANDARD
    .. code-block::

       Set the C++ standard version.

ALPAKA_ACC_<ACC>_ENABLE
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

        Important: Not all accelerator backends are tested, see CI tests.

ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA
    .. code-block::

        Enable lambda support in Alpaka 0.6.x and below for the CUDA accelerator.

ALPAKA_CUDA_EXPT_EXTENDED_LAMBDA
    .. code-block::

        Enable lambda support in Alpaka 0.7.x and later for the CUDA accelerator.