/* Copyright 2021 Hauke Mewes, Simeon Ehrig, Victor
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/reduce/reduce.hpp>
#include <vikunja/mem/iterator/ZipIterator.hpp>

#include <alpaka/alpaka.hpp>

#include <iostream>


template<typename IteratorTuple>
void printTuple(IteratorTuple tuple)
{
    std::cout << "tuple(";
    int index = 0;
    int tupleSize = std::tuple_size<IteratorTuple>{};
    for_each(tuple, [&index, tupleSize](auto &x) { std::cout << *x << (++index < tupleSize ? ", " : ""); });
    std::cout << ")";
}

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
    for_each(std::tuple<Tp...> &, FuncT) // Unused arguments are given no names
    {
    }

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
    for_each(std::tuple<Tp...>& t, FuncT f)
    {
        f(std::get<I>(t));
        for_each<I + 1, FuncT, Tp...>(t, f);
    }

int main()
{
    // Define the accelerator here. Must be one of the enabled accelerators.
    using TAcc = alpaka::AccCpuSerial<alpaka::DimInt<3u>, std::uint64_t>;

    // Types of the data that will be reduced
    using TRed = uint64_t;
    using TRedChar = char;
    using TRedDouble = double;

    // Alpaka index type
    using Idx = alpaka::Idx<TAcc>;
    // Alpaka dimension type
    using Dim = alpaka::Dim<TAcc>;
    // Type of the extent vector
    using Vec = alpaka::Vec<Dim, Idx>;
    // Find the index of the CUDA blockIdx.x component. Alpaka somehow reverses
    // these, i.e. the x component of cuda is always the last value in the vector
    constexpr Idx xIndex = Dim::value - 1u;
    // number of elements to reduce
    const Idx n = static_cast<Idx>(10);
    // create extent
    Vec extent(Vec::all(static_cast<Idx>(1)));
    extent[xIndex] = n;

    // define device, platform, and queue types.
    using DevAcc = alpaka::Dev<TAcc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    // using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::Dev<TAcc>>;
    using PltfHost = alpaka::PltfCpu;
    using DevHost = alpaka::Dev<PltfHost>;
    using QueueAcc = alpaka::Queue<TAcc, alpaka::Blocking>;
    using QueueHost = alpaka::QueueCpuBlocking;

    // Get the host device.
    DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    // Get a queue on the host device.
    QueueHost queueHost(devHost);
    // Select a device to execute on.
    DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    // Get a queue on the accelerator device.
    QueueAcc queueAcc(devAcc);

    // allocate memory both on host and device.
    auto deviceMem(alpaka::allocBuf<TRed, Idx>(devAcc, extent));
    auto hostMem(alpaka::allocBuf<TRed, Idx>(devHost, extent));
    // Fill memory on host with numbers from 0...n-1.
    TRed* hostNative = alpaka::getPtrNative(hostMem);
    for(Idx i = 0; i < n; ++i)
        hostNative[i] = static_cast<TRed>(i + 1);
    // Copy to accelerator.
    alpaka::memcpy(queueAcc, deviceMem, hostMem, extent);
    TRed* deviceNative = alpaka::getPtrNative(deviceMem);
    
    // allocate memory both on host and device.
    auto deviceMemChar(alpaka::allocBuf<TRedChar, Idx>(devAcc, extent));
    auto hostMemChar(alpaka::allocBuf<TRedChar, Idx>(devHost, extent));
    // Fill memory on host with char from 'a' to 'j'.
    TRedChar* hostNativeChar = alpaka::getPtrNative(hostMemChar);
    hostNativeChar[0] = 'a';
    hostNativeChar[1] = 'b';
    hostNativeChar[2] = 'c';
    hostNativeChar[3] = 'd';
    hostNativeChar[4] = 'e';
    hostNativeChar[5] = 'f';
    hostNativeChar[6] = 'g';
    hostNativeChar[7] = 'h';
    hostNativeChar[8] = 'i';
    hostNativeChar[9] = 'j';
    // Copy to accelerator.
    alpaka::memcpy(queueAcc, deviceMemChar, hostMemChar, extent);
    TRedChar* deviceNativeChar = alpaka::getPtrNative(deviceMemChar);

    // allocate memory both on host and device.
    auto deviceMemDouble(alpaka::allocBuf<TRedDouble, Idx>(devAcc, extent));
    auto hostMemDouble(alpaka::allocBuf<TRedDouble, Idx>(devHost, extent));
    // Fill memory on host with double numbers from 10.12...(n-1 + 10.12).
    TRedDouble* hostNativeDouble = alpaka::getPtrNative(hostMemDouble);
    for(Idx i = 0; i < n; ++i)
        hostNativeDouble[i] = static_cast<TRedDouble>(i + 10.12);
    // Copy to accelerator.
    alpaka::memcpy(queueAcc, deviceMemDouble, hostMemDouble, extent);
    TRedDouble* deviceNativeDouble = alpaka::getPtrNative(deviceMemDouble);

    std::cout << "\nTesting zip iterator in host with tuple<uint64_t, char, double>\n\n";

    using IteratorTuple = std::tuple<TRed*, TRedChar*, TRedDouble*>;
    IteratorTuple zipTuple = std::make_tuple(hostNative, hostNativeChar, hostNativeDouble);
    vikunja::mem::iterator::ZipIterator<IteratorTuple> zipIter(zipTuple);

    std::cout << "*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";
    
    std::cout << "*++zipIter: ";
    printTuple(*++zipIter);
    std::cout << "\n*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    std::cout << "*zipIter++: ";
    printTuple(*zipIter++);
    std::cout << "\n*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    zipIter += 6;
    std::cout << "zipIter += 6;\n"
              << "*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    zipIter -= 2;
    std::cout << "zipIter -= 2;\n"
              << "*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    std::cout << "--zipIter: ";
    printTuple(*--zipIter);
    std::cout << "\n*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    std::cout << "zipIter--: ";
    printTuple(*zipIter--);
    std::cout << "\n*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    std::cout << "*(zipIter + 2): ";
    printTuple(*(zipIter + 2));
    std::cout << "\n\n";

    std::cout << "*(nzipIter - 3): ";
    printTuple(*(zipIter - 3));
    std::cout << "\n\n";

    std::cout << "*zipIter[0]: ";
    printTuple(zipIter[0]);
    std::cout << "\n";

    std::cout << "*zipIter[3]: ";
    printTuple(zipIter[3]);
    std::cout << "\n";

    std::cout << "*zipIter[6]: ";
    printTuple(zipIter[6]);
    std::cout << "\n";
    
    std::cout << "*zipIter[9]: ";
    printTuple(zipIter[9]);
    std::cout << "\n\n";



    std::cout << "-----\n\n"
              << "Failed when building for accelerator testing\n\n";

    // std::cout << "Testing accelerator: " << alpaka::getAccName<TAcc>() << " with size: " << n << "\n\n";

    // IteratorTuple deviceZipTuple = std::make_tuple(deviceNative, deviceNativeChar, deviceNativeDouble);
    // vikunja::mem::iterator::ZipIterator<IteratorTuple> deviceZipIter(deviceZipTuple);

    // using IteratorTupleReduceResult = std::tuple<TRed, TRedChar, TRedDouble>;

    // // Use Lambda function for reduction
    // auto sum = [] ALPAKA_FN_HOST_ACC(IteratorTuple const i, IteratorTuple const j)
    // {
    //     IteratorTupleReduceResult tmp = std::make_tuple(*std::get<0>(i) + *std::get<0>(j), *std::get<1>(i), *std::get<2>(i) + *std::get<2>(j));
    //     return tmp;
    // };
    // auto doubleNum = [] ALPAKA_FN_HOST_ACC(IteratorTuple const i)
    // {
    //     IteratorTupleReduceResult tmp = std::make_tuple(2 * *std::get<0>(i), *std::get<1>(i), 2 * *std::get<2>(i));
    //     return tmp;
    // };

    // // TRANSFORM_REDUCE CALL:
    // // Takes the arguments: accelerator device, host device, accelerator queue, size of data, pointer-like to memory,
    // // transform lambda, reduce lambda.
    // auto transformReduceResult = vikunja::reduce::deviceTransformReduce<TAcc>(
    //     devAcc,
    //     devHost,
    //     queueAcc,
    //     n,
    //     deviceZipIter,
    //     doubleNum,
    //     sum);



    return 0;
}
