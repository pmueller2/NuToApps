#pragma once

#include <vector>
#include "ExplicitNystroemNoVelocity.h"

namespace NuTo
{
namespace TimeIntegration
{
template <typename Tstate>
class NY4NoVelocity : public ExplicitNystroemNoVelocity<Tstate>
{
public:
    NY4NoVelocity()
        : NuTo::TimeIntegration::ExplicitNystroemNoVelocity<Tstate>({{0., 0., 0.}, {1. / 8., 0., 0.}, {0., 0.5, 0.}},
                                                                    {1. / 6, 2. / 3., 1. / 6.}, {1. / 6, 1. / 3., 0.},
                                                                    {0., 1. / 2., 1.})
    {
    }
};
}
}
