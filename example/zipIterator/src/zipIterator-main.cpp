/* Copyright 2021 Hauke Mewes, Simeon Ehrig, Victor
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/transform/transform.hpp>
#include <vikunja/mem/iterator/ZipIterator.hpp>

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <variant>

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type forEach(std::tuple<Tp...> &, FuncT) // Unused arguments are given no names
{
}

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type forEach(std::tuple<Tp...>& t, FuncT f)
{
    f(std::get<I>(t));
    forEach<I + 1, FuncT, Tp...>(t, f);
}

template<typename TIteratorTupleVal>
void printTuple(TIteratorTupleVal tuple)
{
    std::cout << "(";
    int index = 0;
    int tupleSize = std::tuple_size<TIteratorTupleVal>{};
    forEach(tuple, [&index, tupleSize](auto &x) { std::cout << x << (++index < tupleSize ? ", " : ""); });
    std::cout << ")";
}

int main()
{
    // Define the accelerator here. Must be one of the enabled accelerators.
    using TAcc = alpaka::AccCpuSerial<alpaka::DimInt<3u>, std::uint64_t>;

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
    auto deviceMem(alpaka::allocBuf<uint64_t, Idx>(devAcc, extent));
    auto hostMem(alpaka::allocBuf<uint64_t, Idx>(devHost, extent));
    // Fill memory on host with numbers from 0...n-1.
    uint64_t* hostNative = alpaka::getPtrNative(hostMem);
    for(Idx i = 0; i < n; ++i)
        hostNative[i] = static_cast<uint64_t>(i + 1);
    // Copy to accelerator.
    alpaka::memcpy(queueAcc, deviceMem, hostMem, extent);
    uint64_t* deviceNative = alpaka::getPtrNative(deviceMem);
    
    // allocate memory both on host and device.
    auto deviceMemChar(alpaka::allocBuf<char, Idx>(devAcc, extent));
    auto hostMemChar(alpaka::allocBuf<char, Idx>(devHost, extent));
    std::vector<char> chars = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j' };
    // Fill memory on host with char from 'a' to 'j'.
    char* hostNativeChar = alpaka::getPtrNative(hostMemChar);
    for(Idx i = 0; i < n; ++i)
    {
        hostNativeChar[i] = chars[i];
    }
    // Copy to accelerator.
    alpaka::memcpy(queueAcc, deviceMemChar, hostMemChar, extent);
    char* deviceNativeChar = alpaka::getPtrNative(deviceMemChar);

    // allocate memory both on host and device.
    auto deviceMemDouble(alpaka::allocBuf<double, Idx>(devAcc, extent));
    auto hostMemDouble(alpaka::allocBuf<double, Idx>(devHost, extent));
    // Fill memory on host with double numbers from 10.12...(n-1 + 10.12).
    double* hostNativeDouble = alpaka::getPtrNative(hostMemDouble);
    for(Idx i = 0; i < n; ++i)
        hostNativeDouble[i] = static_cast<double>(i + 10.12);
    // Copy to accelerator.
    alpaka::memcpy(queueAcc, deviceMemDouble, hostMemDouble, extent);
    double* deviceNativeDouble = alpaka::getPtrNative(deviceMemDouble);

    std::cout << "\nTesting zip iterator in host with tuple<uint64_t, char, double>\n\n";

    using TIteratorTuplePtr = std::tuple<uint64_t*, char*, double*>;
    using TIteratorTupleVal = std::tuple<uint64_t, char, double>;
    TIteratorTuplePtr zipTuple = std::make_tuple(hostNative, hostNativeChar, hostNativeDouble);
    vikunja::mem::iterator::ZipIterator<TIteratorTuplePtr, TIteratorTupleVal> zipIter(zipTuple);

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
    std::cout << "*zipIter += 6;\n"
              << "*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    zipIter -= 2;
    std::cout << "*zipIter -= 2;\n"
              << "*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    std::cout << "*--zipIter: ";
    printTuple(*--zipIter);
    std::cout << "\n*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    std::cout << "*zipIter--: ";
    printTuple(*zipIter--);
    std::cout << "\n*zipIter: ";
    printTuple(*zipIter);
    std::cout << "\n\n";

    // std::cout << "Double the number values of the tuple:\n"
    //           << "zipIter = std::make_tuple(2 * std::get<0>(*zipIter), std::get<1>(*zipIter), 2 * std::get<2>(*zipIter));\n"
    //           << "*zipIter: ";
    // zipIter = std::make_tuple(2 * std::get<0>(*zipIter), std::get<1>(*zipIter), 2 * std::get<2>(*zipIter));
    // printTuple(*zipIter);
    // std::cout << "\n\n";

    std::cout << "*(zipIter + 2): ";
    printTuple(*(zipIter + 2));
    std::cout << "\n\n";

    std::cout << "*(zipIter - 3): ";
    printTuple(*(zipIter - 3));
    std::cout << "\n\n";

    std::cout << "zipIter[0]: ";
    printTuple(zipIter[0]);
    std::cout << "\n";

    std::cout << "zipIter[2]: ";
    printTuple(zipIter[2]);
    std::cout << "\n";

    // std::cout << "zipIter[4] (number values has been doubled): ";
    // printTuple(zipIter[4]);
    // std::cout << "\n";

    // std::cout << "Revert the number values for index 4\n";
    // zipIter = std::make_tuple(std::get<0>(*zipIter) / 2, std::get<1>(*zipIter), std::get<2>(*zipIter) / 2);

    std::cout << "zipIter[4]: ";
    printTuple(zipIter[4]);
    std::cout << "\n";

    std::cout << "zipIter[6]: ";
    printTuple(zipIter[6]);
    std::cout << "\n";

    std::cout << "zipIter[9]: ";
    printTuple(zipIter[9]);
    std::cout << "\n\n"
              << "-----\n\n";

    TIteratorTuplePtr deviceZipTuple = std::make_tuple(deviceNative, deviceNativeChar, deviceNativeDouble);
    vikunja::mem::iterator::ZipIterator<TIteratorTuplePtr, TIteratorTupleVal> deviceZipIter(deviceZipTuple);

    auto deviceMemResult(alpaka::allocBuf<TIteratorTupleVal, Idx>(devAcc, extent));
    auto hostMemResult(alpaka::allocBuf<TIteratorTupleVal, Idx>(devHost, extent));
    TIteratorTupleVal* hostNativeResultPtr = alpaka::getPtrNative(hostMemResult);
    TIteratorTupleVal* deviceNativeResultPtr = alpaka::getPtrNative(deviceMemResult);

    auto doubleNum = [] ALPAKA_FN_HOST_ACC(TIteratorTupleVal const& t)
    {
        return std::make_tuple(2 * std::get<0>(t), std::get<1>(t), 2 * std::get<2>(t));
    };

    vikunja::transform::deviceTransform<TAcc>(
        devAcc,
        queueAcc,
        extent[Dim::value - 1u],
        deviceZipIter,
        deviceNativeResultPtr,
        doubleNum);

    // Copy the data back to the host for validation.
    alpaka::memcpy(queueAcc, hostMemResult, deviceMemResult, extent);

    std::cout << "Testing accelerator: " << alpaka::getAccName<TAcc>() << " with size: " << extent.prod() << "\n"
              << "-----\n";

    bool isTransformSuccess = true;
    for(Idx i = 0; i < n; ++i)
    {
        std::cout << "n=" << i << " | Expected result: ("
                  << 2 * (i + 1) << ", " << chars[i] << ", " << 2 * (i + 10.12)
                  << ") | Actual result: ";
        printTuple(hostNativeResultPtr[i]);

        if((2 * (i + 1) == std::get<0>(hostNativeResultPtr[i])) && 
            (chars[i] == std::get<1>(hostNativeResultPtr[i])) &&
            (2 * (i + 10.12) == std::get<2>(hostNativeResultPtr[i]))
        ) {
            std::cout << " | OK\n";
        }
        else
        {
            std::cout << " | NOT OK\n";
            isTransformSuccess = false;
        }
    }

    if(isTransformSuccess)
        std::cout << "-----\n"
                  << "Transform was successful!\n\n";
    else
        std::cout << "-----\n"
                  << "Transform was NOT successful!\n\n";

    return 0;
}
