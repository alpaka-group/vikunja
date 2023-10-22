#pragma once

#include <vikunja/concept/memVisibility.hpp>

namespace vikunja
{
    template<typename TAcc, typename TMemVisibility = vikunja::concept::get_mem_visibility_type<TAcc>>
    class MemObj
    {
    public:
        using Acc = TAcc;
        using MemVisibility = TMemVisibility;
        // getNativPtr()
        // extend()
        // dim()
    };
} // namespace vikunja
