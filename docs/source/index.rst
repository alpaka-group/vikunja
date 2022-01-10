:orphan:

.. only:: html

  .. image:: logo/vikunja_logo.svg
     :alt: vikunja logo
     :width: 400

.. only:: latex

  .. image:: logo/vikunja_logo.pdf
     :alt: vikunja logo


Vikunja is a performance portable algorithms library defines functions for a variety of purposes that operate on ranges of elements. It supports the execution on multi core CPUs and various GPUs.


vikunja - How to Read This Document
===================================

Generally, **follow the manual pages in-order** to get started. Individual chapters are based on the information of the chapters before. The documentation is separated in three sections. The ``basic`` section explains all concepts of vikunja. In the ``advanced`` you will get more details of certain functionalities. If you want to provide to the project, you should have a look in the ``development`` section.

.. toctree::
   :maxdepth: 1
   :caption: Basic

   basic/introduction.rst
   basic/installation.rst
   basic/algorithm.rst
   basic/operators.rst

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   advanced/cmake.rst

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/docs.rst
   development/ci.rst
   development/apireference.rst