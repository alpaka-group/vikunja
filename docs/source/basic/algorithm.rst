Algorithms
==========

This page provides an overview of all algorithms implemented in vikunja.

All algorithms have the property that the order in which the input elements are accessed is not defined.

Transform
---------

Takes a range of elements as input, applies an unary operator to each element, and writes the result to an output range in the same order.

.. only:: html

  .. image:: images/transform.svg
    :alt: scheme: transform algorithm
    :width: 400

.. only:: latex

  .. image:: images/transform.pdf
    :alt: scheme: transform algorithm

Reduce
------

Takes a range of elements as input and reduces it to a single element via an operator. The operator is a binary operator that takes a subtotal and an element of the input and returns a reduced element. The mathematical operation that the operator applies must be `associative <https://en.wikipedia.org/wiki/Associative_property>`_ and `commutative <https://en.wikipedia.org/wiki/Commutative_property>`_.

.. only:: html

  .. image:: images/reduction.svg
    :alt: scheme: reduce algorithm
    :width: 400

.. only:: latex

  .. image:: images/reduction.pdf
    :alt: scheme: reduce algorithm
