Operators
=========

The operator describes the operation applied to each element of a range of input elements. There are two ways to define an ``operator``: a lambda or a functor.

Lambda
++++++

.. code-block:: c++

    int main(){
        // ...

        // The lambda needs the attribute ALPAKA_FN_HOST_ACC
        auto incTwo = [] ALPAKA_FN_HOST_ACC (){ return val + 1; };

        vikunja::transform::deviceTransform<Acc>(
            devAcc,
            queueAcc,
            extent[Dim::value - 1u],
            deviceNativePtr,
            deviceNativePtr, 
            incTwo // Operator
    );

        return 0;
    }

Functor
+++++++

.. code-block:: c++

    struct IncOne
    {
        // The operator() function needs the attribute ALPAKA_FN_HOST_ACC
        // and must be const 
        template<typename TData>
        ALPAKA_FN_HOST_ACC TData operator()(TData const val) const
        {
            return val + 1;
        }
    };

    int main(){
        // ...

        IncOne incOne;

        vikunja::transform::deviceTransform<Acc>(
            devAcc,
            queueAcc,
            extent[Dim::value - 1u],
            deviceNativePtr,
            deviceNativePtr, 
            incOne // Operator
    );

        return 0;
    }

The functor object must fullfil the requirements of ``std::is_trivially_copyable``. Compared to a lambda, the functor allows the creation of additional member variables and functions.

.. code-block:: c++

    struct MightFunctor {
    private:
        int const m_max;

        ALPAKA_FN_HOST_ACC int crop_value(int const v) const {
            return (v < m_max) ? v : m_max;
        }

    public:

        MightFunctor(int const max) : m_max(max) {}

        ALPAKA_FN_HOST_ACC TData operator()(TData const val) const
        {
            return crop_value(val);
        }
    }

    int main(){
        // ...

        MightFunctor mightFunctor(42);

        // ...

        return 0;

    }

.. warning:: 
    Global functions are not allowed as functor objects for the vikunja ``algorithm`` due to a limitation of the Nvidia CUDA accelerator.


Operator Types
++++++++++++++

Depending on the ``algorithm``, the ``operator`` requires a different number of input arguments. Currently, the vikunja ``algorithm`` requires a unary (one data input) or binary (two data inputs) ``operator``. A vikunja-specific property of the ``operator`` is that they can have an additional ``acc`` argument, which is required for some alpaka-specific functions.

.. code-block:: c++

    struct BinaryOperatorWithoutAccObject
    {
        template<typename TData>
        ALPAKA_FN_HOST_ACC TData operator()(TData const i, TData const j) const
        {
            return i + j;
        }
    };

    struct BinaryOperatorWithAccObject
    {
        template<typename TAcc, typename TData>
        ALPAKA_FN_HOST_ACC TData operator()(TAcc const& acc, TData const i, TData const j) const
        {
            return alpaka::math::max(acc, i, j);
        }

Please read the `alpaka documentation <https://alpaka.readthedocs.io/en/latest/index.html>`_ for more information about available device functions.

.. warning:: 
    The ``acc`` object also allows access to functions that can break the functionality of the vikunja ``algorithm``, such as using the thread index for a manual memory access.
