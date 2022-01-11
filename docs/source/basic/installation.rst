Installation
============

Vikunja builds and installs itself using `CMake <https://cmake.org/>`_. Before you can install and use vikunja, you have to install the only dependency `alpaka <https://github.com/alpaka-group/alpaka>`_. Vikunja supports alpaka from version 0.6 until 0.8. It is recommend to use the latest alpaka version. Alpaka itself has also a single dependency, boost. Read the `alpaka documentation <https://github.com/alpaka-group/alpaka#dependencies>`_ to determine the supported boost version.

**Install alpaka:**

.. code-block:: bash

   # alpaka requires a boost installation, see alpaka documentation
   git clone --depth 1 --branch 0.8.0 https://github.com/alpaka-group/alpaka.git
   mkdir alpaka/build
   cd alpaka/build
   cmake ..
   cmake --install .

**Install vikunja:**

.. code-block:: bash

  git clone https://github.com/alpaka-group/vikunja.git
  mkdir vikunja/build 
  cd vikunja/build  
  cmake ..
  cmake --build .
  cmake --install .

Build Tests and Examples
------------------------

Enable and run the tests:

.. code-block:: bash

   # start in the vikunja project folder
   mkdir build && cd build
   cmake .. -DBUILD_TESTING=ON
   cmake --build .
   ctest

Enable and run an example:

.. code-block:: bash

   # start in the vikunja project folder
   mkdir build && cd build
   cmake .. -DVIKUNJA_BUILD_EXAMPLES=ON
   cmake --build .
   ./example/transform/example_transform


Use vikunja in a CMake project via ``find_package``
---------------------------------------------------

Once vikunja and alpaka are successfully installed, you can use them in your project via the cmake function ``find_packge``.

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.18)
   project(vikunjaProject)

   find_package(vikunja REQUIRED)

   alpaka_add_executable(${CMAKE_PROJECT_NAME} main.cpp)
   target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE vikunja::vikunja)

At the cmake configuration time of your project, you need to enable at least on alpaka accelerator to execute a vikunja function on a processor. The accelerators are enabled via the cmake argument ``-DALPAKA_ACC_<...>_ENABLE=ON``. All existing accelerators can be found `here <https://alpaka.readthedocs.io/en/latest/advanced/cmake.html>`_.

.. code-block:: bash

   # start in the folder of the root CMakeLists.txt
   mkdir build && cd build
   # enable serial CPU and CUDA GPU accelerator
   cmake .. -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON -DALPAKA_ACC_GPU_CUDA_ENABLE=ON
   cmake --build .

By default ``find_package(vikunja)`` runs ``find_package(alpaka)``, if the ``alpaka::alpaka`` target is not already defined.

Use vikunja in a CMake project via ``add_subdirectory``
-------------------------------------------------------

Vikunja also provides cmake integration via ``add_subdirectory``. The `add_subdirectory <https://cmake.org/cmake/help/latest/command/add_subdirectory.html>`_ approach does not require vikunja or alpaka to be installed and allows for easy deployment of a custom vikunja version together with your project.

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.18)
   project(vikunjaProject)

   add_subdirectory(alpaka REQUIRED)
   add_subdirectory(vikunja REQUIRED)

   alpaka_add_executable(${CMAKE_PROJECT_NAME} main.cpp)
   target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE vikunja::vikunja)

.. code-block:: bash

   # start in the folder of the root CMakeLists.txt
   mkdir build && cd build
   # enable OpenMP CPU backend
   cmake .. -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=ON
   cmake --build .

It is also supported to mix the ``find_package`` and ``add_subdirectory`` approaches for vikunja and alpaka.
