//
// Created by hauke on 11.01.19.
//

#ifndef VIKUNJA_ALPAKASETUP_HPP
#define VIKUNJA_ALPAKASETUP_HPP

#include <alpaka/alpaka.hpp>

namespace vikunja {
namespace test {
    template<typename TDim, typename TIdx, template<class, class> class THost,
            template<class, class> class TAcc, typename TQueue>
    struct TestAlpakaSetup {
    public:
        using Dim = TDim;
        using Idx = TIdx;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
        using Host = THost<Dim, Idx>;
        using Acc = TAcc<Dim, Idx>;
        using DevHost = alpaka::dev::Dev<Host>;
        using DevAcc = alpaka::dev::Dev<Acc>;
        using PltfHost = alpaka::pltf::Pltf<DevHost>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using Queue = TQueue;
        using Event = alpaka::event::Event<Queue>;

        DevAcc devAcc;
        DevHost devHost;
        Queue queue;

        TestAlpakaSetup() :  devAcc{alpaka::pltf::getDevByIdx<PltfAcc>(0u)},
            devHost{alpaka::pltf::getDevByIdx<PltfHost>(0u)},
            queue{devAcc} {}
    };
}
}

#endif //VIKUNJA_ALPAKASETUP_HPP
