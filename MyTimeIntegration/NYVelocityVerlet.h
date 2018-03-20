#pragma once

#include <vector>
#include "mechanics/timeIntegration/ExplicitNystroemNoVelocity.h"

namespace NuTo
{
namespace TimeIntegration
{
template <typename Tstate>
class NYVelocityVerlet : public ExplicitNystroemNoVelocity<Tstate>
{
public:
    NYVelocityVerlet()
        : NuTo::TimeIntegration::ExplicitNystroemNoVelocity<Tstate>({{0., 0.}, {0.5, 0.}}, {0.5, 0.5}, {0.5, 0.},
                                                                    {0., 1.})
    {
    }
};
}
}
