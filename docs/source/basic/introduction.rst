Introduction
============

The basic concept of vikunja is to run an ``algorithm`` with an ``operator`` over a range of elements.

* The ``algorithm`` specifies how the elements of one ore more input ranges are accessed and what kind of operator is applied on it. The algorithm also determines the shape of the result. Examples are:

  * **Tranform**: Takes a range of elements as input and returns a range of the same size. **Transform** applies an operator on each element of the input range.
  * **Reduce**: Takes a range of elements as input and returns a single element. The reduce operator takes two elements of the input range, applies an operation to them, and returns a single element. The operator is applied up to the point where only one element remains.
  * For more examples see: :ref:`Algorithm <Algorithm>`
* An ``operator`` describes an algorithm which is applied to one (unary operator) or two (binary operator) elements and returns a result. The following examples assume that **i** is the first and **j** the second input element:

  * **sum**: `return i+j;`
  * **double of an element**: `return 2*i;`

.. literalinclude:: ../../../example/transform/src/transform-main.cpp
   :language: C++
   :caption: transform-main.cpp
