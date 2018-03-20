#pragma once

#include <vector>
#include "mechanics/timeIntegration/ExplicitNystroem.h"

namespace NuTo
{
namespace TimeIntegration
{
template <typename Tstate>
class RK4Nystroem : public ExplicitNystroem<Tstate>
{
public:
    RK4Nystroem()
        : NuTo::TimeIntegration::ExplicitNystroem<Tstate>(
                    {{ 0. ,  0. ,  0. ,  0. },
                     { 0.5,  0. ,  0. ,  0. },
                     { 0. ,  0.5,  0. ,  0. },
                     { 0. ,  0. ,  1. ,  0. }},
                    {{ 0.  ,  0.  ,  0.  ,  0.  },
                     { 0.  ,  0.  ,  0.  ,  0.  },
                     { 0.25,  0.  ,  0.  ,  0.  },
                     { 0.  ,  0.5 ,  0.  ,  0.  }},
                { 1./6,  1./3,  1./3,  1./6},
                {1./6, 1./6,  1./6,  0.        },
                {0. ,  0.5,  0.5,  1. })

    {
    }
};
}
}
