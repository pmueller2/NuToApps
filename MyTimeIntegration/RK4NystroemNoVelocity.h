#pragma once

#include <vector>
#include "mechanics/timeIntegration/ExplicitNystroemNoVelocity.h"

namespace NuTo
{
namespace TimeIntegration
{
template <typename Tstate>
class RK4NystroemNoVelocity : public ExplicitNystroemNoVelocity<Tstate>
{
public:
    RK4NystroemNoVelocity()
        : NuTo::TimeIntegration::ExplicitNystroemNoVelocity<Tstate>(
                  {{}}, {{0., 0., 0., 0.}, {0., 0., 0., 0.}, {0.25, 0., 0., 0.}, {0., 0.5, 0., 0.}},
                  {1. / 6, 1. / 3, 1. / 3, 1. / 6}, {1. / 6, 1. / 6, 1. / 6, 0.}, {0., 0.5, 0.5, 1.})

    {
    }
};
}
}
