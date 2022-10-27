// cppinsight cannot compile the alpaka.hpp header
#define CPPINSIGHT_TEST 0

#include <vikunja/access/MdspanLinear.hpp>

#if CPPINSIGHT_TEST == 0

#    include <vikunja/algorithm/transform.hpp>

#    include <alpaka/alpaka.hpp>

#endif

#include <experimental/mdspan>
#include <iostream>
#include <type_traits>

/**
 * @brief Do the same like std::iota with a n-dimensional mdspan. The iteration order is from the right to the left
 * dimension.
 *
 * @tparam TSpan type of the mdspan
 * @tparam TData type of the functor
 * @param span The mdspan
 * @param index value of the first element
 */
template<typename TSpan, typename TData>
void iota_span(TSpan span, TData index)
{
    static_assert(TSpan::rank() > 0);
    auto functor = [&index](TData input) { return index++; };
    Iterate_mdspan<TSpan::rank()>{}(span, span, functor);
}

#if CPPINSIGHT_TEST == 1

int main()
{
    std::array<int, 12> d;

    // stdex::mdspan m{d.data(), stdex::extents{12}};
    // stdex::mdspan m{d.data(), stdex::extents{2, 6}};
    // stdex::mdspan m{d.data(), stdex::extents{2, 4, 2}};
    stdex::mdspan m{d.data(), stdex::extents{2, 2, 1, 4}};

    iota_span(m, 42);

    for(auto const& v : d)
    {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    return 0;
}

#else

int main()
{
    using Idx = std::uint64_t;
    Idx const num_dims = 5;
    Idx const dim_size = 6;

    using Acc = alpaka::AccCpuSerial<alpaka::DimInt<num_dims>, Idx>;
    // using Acc = alpaka::AccGpuCudaRt<alpaka::DimInt<num_dims>, Idx>;

    auto const devAcc(alpaka::getDevByIdx<Acc>(0u));
    auto const devHost(alpaka::getDevByIdx<alpaka::PltfCpu>(0u));

    using QueueAcc = alpaka::Queue<Acc, alpaka::Blocking>;
    QueueAcc queueAcc(devAcc);


    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Data = uint64_t;

    using Vec = alpaka::Vec<Dim, Idx>;
    Vec extent(Vec::all(num_dims));
    for(Idx dim = 0; dim < num_dims; ++dim)
    {
        extent[dim] = static_cast<Idx>(dim_size);
    }


    auto deviceMem(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    auto deviceSpan = alpaka::getMdSpan(deviceMem);
    auto hostMem(alpaka::allocBuf<Data, Idx>(devHost, extent));
    Data* hostNativePtr = alpaka::getPtrNative(hostMem);
    auto hostSpan = alpaka::getMdSpan(hostMem);

    iota_span(hostSpan, 1);

    auto doubleNum = [] ALPAKA_FN_HOST_ACC(Data const& i) { return 2 * i; };


    alpaka::memcpy(queueAcc, deviceMem, hostMem, extent);

    vikunja::device::transform<Acc>(devAcc, queueAcc, deviceSpan, deviceSpan, doubleNum);


    // Copy the data back to the host for validation.
    alpaka::memcpy(queueAcc, hostMem, deviceMem, extent);

    Data resultSum = std::accumulate(hostNativePtr, hostNativePtr + extent.prod(), 0);

    Data expectedResult = (extent.prod() * (extent.prod() + 1));

    std::cout << "Testing accelerator: " << alpaka::getAccName<Acc>() << " with size: " << extent.prod() << "\n";
    if(expectedResult == resultSum)
    {
        std::cout << "Transform was successful!\n";
    }
    else
    {
        std::cout << "Transform was not successful!\n"
                  << "expected result: " << expectedResult << "\n"
                  << "actual result: " << resultSum << std::endl;
    }

    return 0;
}
#endif