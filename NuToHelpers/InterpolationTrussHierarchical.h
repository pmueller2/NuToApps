#pragma once
#include <eigen3/Eigen/Core>
#include "nuto/mechanics/interpolation/InterpolationSimple.h"
#include "nuto/math/Legendre.h"
#include "nuto/base/Exception.h"
#include "nuto/math/shapes/Line.h"

namespace NuTo
{
class InterpolationTrussHierarchical : public InterpolationSimple
{
public:
    InterpolationTrussHierarchical(int degree) : mDegree(degree)
    {
        if (mDegree<1)
            throw Exception(__PRETTY_FUNCTION__,"Degree must be at least 1.");
    }

    std::unique_ptr<InterpolationSimple> Clone() const override
    {
        return std::make_unique<InterpolationTrussHierarchical>(*this);
    }

    ShapeFunctions GetShapeFunctions(const NaturalCoords& coords) const override
    {
        double x = coords[0];
        Eigen::VectorXd shapes(mDegree+1);
        shapes[0] = (1.-x)*0.5;
        shapes[1] = (1.+x)*0.5;
        for (int i=2; i<=mDegree;i++)
        {
            shapes[i] = Math::Polynomial::Legendre(i,x) - Math::Polynomial::Legendre(i-2,x);
            shapes[i] /= sqrt(2.*(2*i-1));
        }
        return shapes;
    }

    DerivativeShapeFunctionsNatural GetDerivativeShapeFunctions(const NaturalCoords& coords) const override
    {
        double x = coords[0];
        Eigen::VectorXd derivativeShapes(mDegree+1);
        derivativeShapes[0] = -0.5;
        derivativeShapes[1] = 0.5;
        for (int i=2; i<=mDegree;i++)
        {
            derivativeShapes[i] = Math::Polynomial::Legendre(i,x,1) - Math::Polynomial::Legendre(i-2,x,1);
            derivativeShapes[i] /= sqrt(2.*(2*i-1));
        }
        return derivativeShapes;
    }

    NaturalCoords GetLocalCoords(int nodeId) const override
    {
        switch (nodeId) {
        case 0: return Eigen::VectorXd::Constant(1, -1.);
        case 1: return Eigen::VectorXd::Constant(1, +1.);
        default: throw Exception(__PRETTY_FUNCTION__,"No Coordinate known for node with Id " + std::to_string(nodeId) + ".");
        }
    }

    int GetNumNodes() const override
    {
        return (mDegree+1);
    }

    const Shape& GetShape() const override
     {
         return mShape;
     }

private:
    int mDegree;
    Line mShape;
};
} /* NuTo */
