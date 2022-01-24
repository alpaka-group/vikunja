Testing and Benchmarking
========================

Vikunja offers different types of tests. The source code is tested via unit and integration tests with `Cacht2 <https://github.com/catchorg/Catch2/tree/v2.x>`_. The CMake code is tested with integration tests and custom scripts.

Source Code Tests
-----------------

Before you start writing source code tests, you should read the `Catch2 documentation <https://github.com/catchorg/Catch2/blob/v2.x/docs/tutorial.md#top>`_. Tests written with Catch2 are standalone application. Therefore, they have their own source code files and ``CMakeLists.txt`` files located in the ``test/unit`` and ``test/integ`` folders. If you set the CMake argument ``-DBUILD_TESTING=ON``, the executable files of the tests will be built automatically. All test cases are registered via the CMake function ``add_test``. Therefore, you can automatically run all tests in the build folder with the ``ctest`` command:

.. code-block:: bash

    mkdir build && cd build
    cmake .. -DBUILD_TESTING=ON
    cmake --build .
    ctest

For more CMake arguments for the tests, see the :ref:`CMake section <cmake-test>`.

If you only want to run a single test, you can run the test executable directly. All test executables are located in the ``<build_folder>/tests``. It is also possible to run the executable with the ``--help`` flag to show additional options. For example, the ``-s`` flag displays additional information created with the Catch2 function ``INFO()``.

.. code-block:: bash

    mkdir build && cd build
    cmake .. -DBUILD_TESTING=ON
    cmake --build .
    # display extra test options
    test/integ/reduce/test_reduce --help
    # run test with extra output
    test/integ/reduce/test_reduce -s

CMake Tests
-----------

The CMake integration tests check whether vikunja can be used correctly in another project via the CMake functions ``find_package()``) or ``add_subdirectory``. The CI contains test jobs for creating projects that use the vikunja library. The job names start with ``integration``. All associated files for the tests are in ``script/integration_test``.

CXX Test
++++++++

There is a special Catch2 test that tests the vikunja CMake to see if the C++ standard is set correctly. The name of the test is ``test_cxx``. The test compares the C++ standard set by the compiler with an expected standard passed as an argument. By default, ``ctest`` automatically passes the expected C++ standard depending on the CMake variable ``ALPAKA_CXX_STANDARD``. If you run the test manually, you must pass it yourself:

.. code-block:: bash

    # expected, that the code was compiled with C++ 17
    test/unit/cxx/test_cxx --cxx 17


Benchmarks
----------

Vikunja uses `Catch2 benchmark <https://github.com/catchorg/Catch2/blob/v2.x/docs/benchmarks.md#top>`_ to automatically run benchmarks. By default, the benchmarks are not enabled. To enable the benchmarks, the CMake arguments ``-DBUILD_TESTING=ON -DVIKUNJA_ENABLE_BENCHMARKS=ON`` must be set. The benchmarks are created automatically and can be run with ``ctest``. As with the tests, you can run a particular benchmark directly from the executable file, e.g. ``test/benchmarks/transform/bench_vikunja_transform``. All benchmark executables are located in ``<build_folder>/test/benchmarks``.

.. tip::

    If you run ``bechmark_exe --help``, you get benchmark specific options.
