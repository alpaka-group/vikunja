Continuous Integration
======================

For automatic testing we use ``GitLab CI``. ``GitLab CI`` allows us to run runtime tests on GPUs and CPU architectures other than x86, like ARM or IBM POWER. The GitHub repository is mirrored on https://gitlab.com/hzdr/crp/vikunja . Every commit or pull request is automatically mirrored to GitLab and triggers the CI. The configuration of the ``GitLab CI`` is stored in the file ``.gitlab-ci.yml``. ``GitLab CI`` uses containers which are already prepared for the tests. The containers are built in an `extra repository <https://gitlab.com/hzdr/crp/alpaka-group-container>`_ and contain all dependencies for vikunja. All available containers can be found `here <https://gitlab.com/hzdr/crp/alpaka-group-container/container_registry>`_. The scripts to build vikunja and run the tests are located at ``script/``.

.. figure:: images/arch_gitlab_mirror.svg
   :alt: Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners

   Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners

To change how the tests are built and executed, modify the code in the `vikunja repository <https://github.com/alpaka-group/vikunja>`_. If the container environment with the dependencies needs to be changed, please open an issue or contribute to `alpaka-group-container repository <https://gitlab.com/hzdr/crp/alpaka-group-container/container_registry>`_.
