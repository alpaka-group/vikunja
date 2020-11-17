#include <vikunja/mem/iterator/BaseIterator.hpp>
#include <vikunja/mem/iterator/PolicyBasedBlockIterator.hpp>
#include <alpaka/alpaka.hpp>
#include <catch2/catch.hpp>
#include <vector>

using Type = uint64_t;
using IType = Type*;
using Idx = std::size_t;
using namespace vikunja::mem::iterator;

std::vector<Type> generateIndexVector(Idx size)
{
    std::vector<Type> data(size);
    for(Idx i = 0; i < size; ++i)
    {
        data[i] = i;
    }
    return data;
}

TEST_CASE("BaseIterator", "[iterator]")
{
    constexpr Idx size = 64;
    std::vector<Type> testData{generateIndexVector(size)};

    BaseIterator<IType> zeroFirst(testData.data(), 0, size);
    BaseIterator<IType> zeroSecond(testData.data(), 0, size);
    BaseIterator<IType> one(testData.data(), 1, size);
    BaseIterator<IType> copyOfZeroFirst(zeroFirst);


    REQUIRE(zeroFirst == zeroSecond);
    REQUIRE(zeroSecond == zeroFirst);
    REQUIRE(zeroFirst != one);
    REQUIRE(one != zeroFirst);
    REQUIRE(zeroFirst < one);
    REQUIRE(one > zeroFirst);
    REQUIRE_FALSE(zeroFirst < zeroSecond);
    REQUIRE_FALSE(zeroFirst > zeroSecond);
    REQUIRE(zeroFirst <= zeroSecond);
    REQUIRE(zeroSecond <= zeroFirst);
    REQUIRE(zeroFirst <= one);
    REQUIRE(*zeroFirst == 0);
    REQUIRE(*one == 1);
    REQUIRE(copyOfZeroFirst == zeroFirst);

    *zeroFirst = 2;
    REQUIRE(*zeroFirst == 2);
    REQUIRE(*zeroSecond == 2);
};

template<typename MemAccessPolicy>
struct TestPolicyBasedBlockIteratorKernel
{
};

template<>
struct TestPolicyBasedBlockIteratorKernel<vikunja::mem::iterator::policies::LinearMemAccessPolicy>
{
    template<typename TAcc, typename TIdx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, Type* data, TIdx const& n) const
    {
    }
};

template<typename TAcc>
struct TestPolicyBasedBlockIterator
{
    template<typename MemAccessPolicy>
    void operator()()
    {
        using Idx = alpaka::idx::Idx<TAcc>;
        using Dim = alpaka::dim::Dim<TAcc>;

        constexpr Idx n = 65;
        constexpr Idx blocksPerGrid = 2;
        constexpr Idx threadsPerBlock = 1;
        constexpr Idx elementsPerThread = n / 2 + 1;

        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using QueueAcc = alpaka::queue::QueueCpuBlocking;
        // alpaka::test::queue::DefaultQueue<alpaka::dev::Dev<TAcc>>;
        using PltfHost = alpaka::pltf::PltfCpu;
        using DevHost = alpaka::dev::Dev<PltfHost>;
        using QueueHost = alpaka::queue::QueueCpuBlocking;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
        // Get the host device.
        DevHost devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
        // Get a queue on the host device.
        QueueHost queueHost(devHost);
        // Select a device to execute on.
        DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
        // Get a queue on the accelerator device.
        QueueAcc queueAcc(devAcc);
        WorkDiv workdiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    }
};

TEST_CASE("PolicyBasedBlockIterator", "[iterator]")
{
    //    constexpr Idx size = 64;

    SECTION("LinearMemAccessPolicy")
    {
    }
}