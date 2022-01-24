/* Copyright 2021 Hauke Mewes, Simeon Ehrig, Victor
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/reduce/reduce.hpp>
#include <vikunja/mem/iterator/PermutationIterator.hpp>

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <list>

int main()
{
    // Define the accelerator here. Must be one of the enabled accelerators.
    //using TAcc = alpaka::AccGpuCudaRt<alpaka::DimInt<3u>, std::uint64_t>;
    using TAcc = alpaka::AccCpuSerial<alpaka::DimInt<3u>, std::uint64_t>;

    // Type of the data that will be reduced
    using TRed = uint64_t;

    // Alpaka index type
    using Idx = alpaka::Idx<TAcc>;
    // Alpaka dimension type
    using Dim = alpaka::Dim<TAcc>;
    // number of elements to reduce
    const Idx n = static_cast<Idx>(3);

    /*
    // define device, platform, and queue types.
    using DevAcc = alpaka::Dev<TAcc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    // using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::Dev<TAcc>>;
    using PltfHost = alpaka::PltfCpu;
    using DevHost = alpaka::Dev<PltfHost>;
    using QueueAcc = alpaka::Queue<TAcc, alpaka::Blocking>;

    // Get the host device.
    DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    // Select a device to execute on.
    DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    // Get a queue on the accelerator device.
    QueueAcc queueAcc(devAcc);

    typedef std::vector<TRed> element_range_type;
    typedef std::vector<TRed> index_type;

    // Use Lambda function for reduction
    // auto sum = [] ALPAKA_FN_HOST_ACC(vikunja::mem::iterator::PermutationIterator<ValIterator, IdxIterator> i, vikunja::mem::iterator::PermutationIterator<ValIterator, IdxIterator> j) { return i + j; };
    // auto doubleNum = [] ALPAKA_FN_HOST_ACC(TRed const i) { return 2 * i; };
    std::cout << "Testing accelerator: " << alpaka::getAccName<TAcc>() << " with size: " << n << "\n"
              << "Testing permutation iterator with value: 10\n";
    //*/

    std::vector<TRed> values{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    std::vector<TRed> indices{ 5, 7, 10 };

    // Create the permutation iterator
    typedef vikunja::mem::iterator::PermutationIterator<element_range_type::iterator, index_type::iterator> permutation_type;

    permutation_type permutationIter(values.begin(), indices.begin(), n);

    std::cout << "Permutation result:\n";

    auto it = permutationIter.begin();

    // for (int i = 0; i < n; ++i)
    // {
    //     std::cout << indices[i] << "\t: " << it[i] << "\n";
    // }
    
    std::cout << "*++permutationIter: ";
    std::cout << *++permutationIter;
    std::cout << "\n*permutationIter: ";
    std::cout << *permutationIter;
    std::cout << "\n\n";

    std::cout << "*permutationIter++: ";
    std::cout << *permutationIter++;
    std::cout << "\n*permutationIter: ";
    std::cout << *permutationIter;
    std::cout << "\n\n";

    permutationIter += 6;
    std::cout << "permutationIter += 6;\n"
              << "*permutationIter: ";
    std::cout << *permutationIter;
    std::cout << "\n\n";

    permutationIter -= 2;
    std::cout << "permutationIter -= 2;\n"
              << "*permutationIter: ";
    std::cout << *permutationIter;
    std::cout << "\n\n";

    std::cout << "--permutationIter: ";
    std::cout << *--permutationIter;
    std::cout << "\n*permutationIter: ";
    std::cout << *permutationIter;
    std::cout << "\n\n";

    std::cout << "permutationIter--: ";
    std::cout << *permutationIter--;
    std::cout << "\n*permutationIter: ";
    std::cout << *permutationIter;
    std::cout << "\n\n";

    std::cout << "*(permutationIter + 2): ";
    std::cout << *(permutationIter + 2);
    std::cout << "\n\n";

    std::cout << "*(permutationIter - 3): ";
    std::cout << *(permutationIter - 3);
    std::cout << "\n\n";

    std::cout << "*permutationIter[0]: ";
    std::cout << permutationIter[0];
    std::cout << "\n";

    std::cout << "*permutationIter[3]: ";
    std::cout << permutationIter[3];
    std::cout << "\n";

    // REDUCE CALL:
    // Takes the arguments: accelerator device, host device, accelerator queue, size of data, pointer-like to memory,
    // reduce lambda.
    // Idx reduceResult = vikunja::reduce::deviceReduce<TAcc,
    //     vikunja::workdiv::BlockBasedPolicy<TAcc>,
    //     vikunja::mem::iterator::MemAccessPolicy<TAcc>,
    //     TFunc,
    //     TInputIterator,
    //     DevAcc,
    //     DevHost,
    //     QueueAcc,
    //     Idx,
    //     TOperator = vikunja::operators::BinaryOp<
    //         TAcc,
    //         TFunc,
    //         typename std::iterator_traits<TInputIterator>,
    //         typename std::iterator_traits<TInputIterator>>,
    //         typename TRed = typename TOperator::TRed>(
    // // Idx reduceResult = vikunja::reduce::deviceReduce<TAcc>(
    //     devAcc, 
    //     devHost, 
    //     queueAcc, 
    //     n, 
    //     permutationIterBegin,
    //     sum);

    // // check reduce result
    // auto expectedResult = n * 10;
    // std::cout << "Expected reduce result: " << expectedResult << ", real result: " << reduceResult << "\n";

    // TRANSFORM_REDUCE CALL:
    // Takes the arguments: accelerator device, host device, accelerator queue, size of data, pointer-like to memory,
    // transform lambda, reduce lambda.
    // Idx transformReduceResult = vikunja::reduce::deviceTransformReduce<TAcc>(
    //     devAcc,
    //     devHost,
    //     queueAcc,
    //     n,
    //     permutationIterBegin,
    //     doubleNum,
    //     sum);

    // // check transform result
    // auto expectedTransformReduce = expectedResult * 2;
    // std::cout << "Expected transform_reduce result: " << expectedTransformReduce
    //           << ", real result: " << transformReduceResult << "\n";

    return 0;
}
