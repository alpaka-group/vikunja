Continuous Integration
======================

For automatic testing we use ``GitLab CI``. ``GitLab CI`` allows us to run runtime tests on GPUs and CPU architectures other than x86, like ARM or IBM POWER. The GitHub repository is mirrored on https://gitlab.com/hzdr/crp/vikunja . Every commit or pull request is automatically mirrored to GitLab and triggers the CI. The configuration of the ``GitLab CI`` is stored in the file ``.gitlab-ci.yml``.

Most of the jobs are generated automatically. For more information, see the section :ref:`The Job Generator`.

It is also possible to define custom jobs, see :ref:`Custom jobs`.

``GitLab CI`` uses containers which are already prepared for testing. The containers are built in an `extra repository <https://gitlab.hzdr.de/crp/alpaka-group-container>`_ and contain most of the dependencies for vikunja. see the section :ref:`The Container Registry` for more information

All CI related scripts are located at ``ci/``.

.. figure:: images/arch_gitlab_mirror.svg
   :alt: Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners

   Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners

The Container Registry
----------------------

Vikunja uses containers in which as many dependencies as possible are already installed to save job execution time. The available containers can be found `here <https://gitlab.hzdr.de/crp/alpaka-group-container/container_registry>`_. Each container provides a tool called ``agc-manager`` to check if a software is installed. The documentation for ``agc-manager`` can be found `here <https://gitlab.hzdr.de/crp/alpaka-group-container/-/tree/master/tools>`_. A common way to check if a software is already installed is to use an ``if else statement``. If a software is not installed yet, you can install it every time at job runtime.

.. code-block:: bash

 if agc-manager -e boost@${VIKUNJA_CI_BOOST_VER} ; then
   export VIKUNJA_CI_BOOST_ROOT=$(agc-manager -b boost@${VIKUNJA_CI_BOOST_VER})
 else
   # install boost
 fi

This statement installs a specific boost version until the boost version is pre-installed in the container. To install a specific software permanently in the container, please open an issue in the `alpaka-group-container repository <https://gitlab.hzdr.de/crp/alpaka-group-container/-/issues>`_.

The Job Generator
-----------------

Vikunja supports a large number of different compilers with different versions and build configurations. To manage this large set of possible test cases, we use a job generator that generates the CI jobs for the different compiler and build configuration combinations. The jobs do not cover all possible combinations, as it would be too much to run the entire CI pipeline in a reasonable amount of time. Instead, the job generator uses `pairwise testing <https://en.wikipedia.org/wiki/All-pairs_testing>`_.

The stages of the job generator are:

1. Assembles all input parameters depending on the `version.py <https://github.com/alpaka-group/vikunja/blob/master/ci/job_generator/versions.py>`_.
2. Create the sparse combination matrix. Different rules are applied during the process to avoid invalid combinations, e.g. compiling the CUDA backend with the HIP compiler.
3. Shuffle the jobs. By default, the pairwise generator has a systematic way of generating the jobs, which is why two consecutive jobs are only slightly different in the results list. Shuffling increases the variety among the first jobs and the chance to find an error early.
4. Manually reorder jobs. Manually place some jobs at the beginning to increase the chance of finding an error early.
5. Generate the yaml code for each combination.
6. Add custom jobs. The yaml code is loaded from files.
7. Distribute the jobs in waves. Jobs are started in waves because we have a limited number of resources. The waves allow a fair distribution of resources between all projects using the HZDR CI Runner.
8. Write jobs to a yaml file.

The job generator is located at `ci/job_generator/ <https://github.com/alpaka-group/vikunja/blob/master/ci/job_generator/>`_. The code is split into two parts. One part is vikunja-specific and stored in this repository. The other part is valid for all alpaka-based projects and stored in the `alpaka-job-coverage library <https://pypi.org/project/alpaka-job-coverage/>`_.

Run Job Generator Offline
+++++++++++++++++++++++++

First you need to install the dependencies. It is highly recommended to use a virtual environment. You can create one for example with the `venv <https://docs.python.org/3/library/venv.html>`_-Python module or with `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. Once you have created a virtual environment, you should activate it and install the Python packages via:

.. code-block:: bash

 pip install -r ci/job_generator/requirements.txt

After installing the Python package, you can simply run the job generator via:

.. code-block:: bash

 # 3.0 is the version of the docker container image
 # run `python ci/job_generator/job_generator.py --help` to see more options
 python ci/job_generator/job_generator.py 3.0

The generator creates a ``jobs.yaml`` in the current directory with all job combinations.

Develop new Feature for the alpaka-job-coverage Library
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

Sometimes one needs to implement a new function or fix a bug in the alpaka-job-coverage library while they are implementing a new function or fixing a bug in the vikunja job generator. Affected filter rules can be recognized by the fact that they only use parameters defined in this `globals.py <https://github.com/alpaka-group/alpaka-job-matrix-library/blob/main/src/alpaka_job_coverage/globals.py>`_.

The following steps explain how to set up a development environment for the alpaka-job-coverage library and test your changes with the vikunja job generator.

We strongly recommend using a Python virtual environment.

.. code-block:: bash

 # if not already done, clone repositories
 git clone https://github.com/alpaka-group/alpaka-job-matrix-library.git
 git clone https://github.com/alpaka-group/vikunja.git

 cd alpaka-job-matrix-library
 # link the files from the alpaka-job-matrix-library project folder into the site-packages folder of your environment
 # make the package available in the Python interpreter via `import alpaka_job_coverage`
 # if you change a src file in the folder, the changes are immediately available (if you use a Python interpreter instance, you have to restart it)
 python setup.py develop
 cd ..
 cd vikunja
 pip install -r ci/job_generator/requirements.txt

Now you can simply run the vikunja job generator. If you change the source code in the project folder alpaka-job-matrix-library, it will be immediately available for the next generator run.

Custom jobs
-----------

You can create custom jobs that are defined as a yaml file. The job yaml files are normally stored in ``ci/custom_jobs/`` and included in the ``add_custom_jobs()`` function in ``ci/custom_jobs/custom_job.py``. The custom jobs are added to the same job list as the generated jobs and distributed to the waves.
