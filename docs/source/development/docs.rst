Documentation
=============

Vikunja uses two tools to create documentation: `Sphinx Doc <https://www.sphinx-doc.org/en/master/>`_ and `Doxygen <https://www.doxygen.nl/index.html>`_. Sphinx is used to write this general documentation and Doxygen generates the API documentation.

Building
++++++++

The vikunja documentation requires Python 3 and Doxygen. It is also recommended to use a Python environment.

.. code-block:: bash

    cd docs
    pip install -r requirements.txt
    make html
    # open it with a web browser, e.g. Firefox
    firefox build/html/index.html

Doxygen documentation is built automatically with Sphinx Doc. If you want to built the documentation manually, follow the steps below:

.. code-block:: bash

    cd docs
    doxygen Doxyfile
    firefox build/doxygen/html