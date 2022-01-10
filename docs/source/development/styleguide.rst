Style Guide
===========

Vikunja uses the style guide of the ``Computational Radiation Physics`` group. The style guide is defined in a ``.clang-format`` file and can be found `here <https://github.com/ComputationalRadiationPhysics/contributing/tree/master/formatting-tools>`_. For correct application of the ``.clang-format`` file, please use ``clang-format`` version ``12.0.1``.

Format a single file with: 
  
.. code-block:: bash
  
    clang-format -i --style=file <sourcefile>``

Format all files in the project:

.. code-block:: bash

   find example include test -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i --style=file 